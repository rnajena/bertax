import torch
import tensorflow as tf
from bertax.utils import load_bert, get_token_dict, seq2tokens, parse_fasta
import numpy as np
from transformers import BertModel, BertConfig
import bertviz
import json
import argparse
import webbrowser
from logging import warning
import pkg_resources
import os.path
import re


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize BERT model with provieded sequence')
    parser.add_argument('fasta')
    parser.add_argument('-a', type=int, help='sequence start in nt (Default: 0)',
                        default=0)
    parser.add_argument('-n', type=int, help='maximum sequence size in nt (Default: 500)',
                        default=500)
    parser.add_argument('--mode', choices=['head', 'model'],
                        default='head', help='attention head view or model view (Default: head)')
    parser.add_argument('--dont_open', action='store_true',
                        help='Don\'t open generated HTML view in Browser')
    return parser.parse_args()

def keras2torch(kmodel,
                params={'embed_dim': 250, 'seq_len': 502, 'transformer_num': 12,
                        'head_num': 5, 'feed_forward_dim': 1024,
                        'dropout_rate': 0.05, 'vocab_size': 69}):
    tmodel = BertModel(BertConfig(vocab_size=params['vocab_size'],
                                  hidden_size=params['embed_dim'],
                                  num_attention_heads=params['head_num'],
                                  num_hidden_layers=params['transformer_num'],
                                  intermediate_size=params['feed_forward_dim'],
                                  hidden_dropout_prob=params['dropout_rate'],
                                  attention_probs_dropout_prob=params['dropout_rate'],
                                  max_position_embeddings=params['seq_len'],
                                  layer_norm_eps=tf.keras.backend.epsilon() * tf.keras.backend.epsilon()))
    # set torch model tensors to the ones from the keras model
    td = {t[0]: t[1] for t in tmodel.named_parameters()}
    kd = {t.name: t for t in kmodel.weights}
    def set_tensor(tname, karray):
        assert (tshape:=td[tname].detach().numpy().shape) == (
            kshape:=karray.shape), f'{tname} has incompatible shape: {tshape} != {kshape}'
        with torch.no_grad():
            td[tname].data = torch.nn.Parameter(torch.Tensor(karray))
    # 1 INPUT
    t_pfix = 'embeddings.'
    k_pfix = 'Embedding-'
    # set_tensor(t_pfix + 'position_ids', td[t_pfix + 'position_ids']) # don't change
    set_tensor(t_pfix + 'word_embeddings.weight', kd[k_pfix + 'Token/embeddings:0'].numpy())
    set_tensor(t_pfix + 'position_embeddings.weight', kd[k_pfix + 'Position/embeddings:0'].numpy())
    set_tensor(t_pfix + 'token_type_embeddings.weight', kd[k_pfix + 'Segment/embeddings:0'].numpy())
    set_tensor(t_pfix + 'LayerNorm.weight', kd[k_pfix + 'Norm/gamma:0'].numpy())
    set_tensor(t_pfix + 'LayerNorm.bias', kd[k_pfix + 'Norm/beta:0'].numpy())
    # 2 LAYERS
    for i in range(params['transformer_num']):
        t_pfix_l = f'encoder.layer.{i}.'
        k_pfix_l = f'Encoder-{i+1}-'
        # SELF-ATTENTION
        # NOTE: (embed_dim x embed_dim) matrices have to be transposed!
        t_pfix = t_pfix_l + 'attention.'
        k_pfix = k_pfix_l + f'MultiHeadSelfAttention/Encoder-{i+1}-MultiHeadSelfAttention_'
        set_tensor(t_pfix + 'self.query.weight', kd[k_pfix + 'Wq:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.query.bias', kd[k_pfix + 'bq:0'].numpy())
        set_tensor(t_pfix + 'self.key.weight', kd[k_pfix + 'Wk:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.key.bias', kd[k_pfix + 'bk:0'].numpy())
        set_tensor(t_pfix + 'self.value.weight', kd[k_pfix + 'Wv:0'].numpy().transpose())
        set_tensor(t_pfix + 'self.value.bias', kd[k_pfix + 'bv:0'].numpy())
        set_tensor(t_pfix + 'output.dense.weight', kd[k_pfix + 'Wo:0'].numpy().transpose())
        set_tensor(t_pfix + 'output.dense.bias', kd[k_pfix + 'bo:0'].numpy())
        # NORM
        t_pfix = t_pfix_l + 'attention.output.LayerNorm.'
        k_pfix = k_pfix_l + f'MultiHeadSelfAttention-Norm/'
        set_tensor(t_pfix + 'weight', kd[k_pfix + 'gamma:0'].numpy())
        set_tensor(t_pfix + 'bias', kd[k_pfix + 'beta:0'].numpy())
        # FF
        t_pfix = t_pfix_l + ''
        k_pfix = k_pfix_l + 'FeedForward'
        set_tensor(t_pfix + 'intermediate.dense.weight',
                   kd[k_pfix + f'/Encoder-{i+1}-FeedForward_W1:0'].numpy().transpose())
        set_tensor(t_pfix + 'intermediate.dense.bias', kd[k_pfix + f'/Encoder-{i+1}-FeedForward_b1:0'].numpy())
        set_tensor(t_pfix + 'output.dense.weight',
                   kd[k_pfix + f'/Encoder-{i+1}-FeedForward_W2:0'].numpy().transpose())
        set_tensor(t_pfix + 'output.dense.bias', kd[k_pfix + f'/Encoder-{i+1}-FeedForward_b2:0'].numpy())
        set_tensor(t_pfix + 'output.LayerNorm.weight', kd[k_pfix + '-Norm/gamma:0'].numpy())
        set_tensor(t_pfix + 'output.LayerNorm.bias', kd[k_pfix + '-Norm/beta:0'].numpy())
    # 3 OUTPUT (before class)
    set_tensor('pooler.dense.weight', kd['NSP-Dense/kernel:0'].numpy().transpose())
    set_tensor('pooler.dense.bias', kd['NSP-Dense/bias:0'].numpy())
    return tmodel

def js_data(tmodel, in_ids, tokend):
    out = tmodel(input_ids=torch.tensor(np.array([in_ids]), dtype=torch.long), output_attentions=True)
    attn = bertviz.util.format_attention(out[-1]).tolist()
    tokens = list(map({v: k for k, v in tokend.items()}.__getitem__, in_ids))
    return {'attn': attn, 'left_text': tokens, 'right_text': tokens}

def ins_json(tmodel, in_ids, tokend, out_handle):
    out_handle.write('PYTHON_PARAMS = ');
    json.dump({'default_filter': 'all', 'root_div_id': 'bert_viz',
               'display_mode': 'dark',
               'attention': {'all': js_data(tmodel, in_ids, tokend)}},
              out_handle, indent=2)

def get_view_js_lines(mode):
    # NOTE: due to an incompatibility in the only available pypi version of bertviz, the
    # script has to be read in and inserted into the HTML
    # TODO: in a future, compatible version, <script src='{path}'></script> would suffice
    view_js_location = os.path.join(bertviz.__path__[0], f'{mode}_view.js')
    return ('<script>' + re.sub(r'require.config\(.[^\)]+\);', '', open(view_js_location).read())
            + '</script>')

def main():
    args = parse_arguments()
    fasta = parse_fasta(args.fasta)
    if (len(fasta) > 1):
        warning(f'Only one sequence supported, taking first one {fasta[0].id}')
    seq = fasta[0].seq
    tokend = get_token_dict()
    # load & convert model
    m = load_bert(pkg_resources.resource_filename(
        'bertax', 'resources/big_trainingset_all_fix_classes_selection.h5'))
    # size visualize has to be smaller than the maximum possible model size (* 3) and the sequence size
    model_len = m.layers[0].input_shape[0][1]
    args.n = min(min(args.n, np.ceil(len(seq)).astype(int)), (model_len - 1) * 3)
    # converted 🤗 transformers model
    tm = keras2torch(m)
    _ = tm.eval()                   # toggle evaluation mode, no output
    # whole sequence to tokens
    ins = seq2tokens(seq[args.a:(args.a + args.n)], tokend,
                     np.ceil(args.n / 3 + 1).astype(int))
    # dump json
    ins_json(tm, ins[0], tokend, open('view_data.js', 'w'))
    # get {model|head}_view script
    view_js_lines = get_view_js_lines(args.mode)
    # create HTML
    html = """
    <html>
      <body>
        <div id='bert_viz'>
          <span style="user-select:none">
            Layer: <select id="layer"></select>
          </span>
          <div id='vis'></div>
        </div>
        <script src="https://requirejs.org/docs/release/2.3.6/minified/require.js"></script>
        <script>
          require.config({
              paths: {
                  d3: 'https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min',
                  jquery: 'https://ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
              }
          });
        </script>
        <script src="view_data.js"></script>
        %s
      </body>
    </html>"""%(view_js_lines)
    with open('view_model.html', 'w') as f:
        f.write(html)
    if (not args.dont_open):
        webbrowser.open('view_model.html')

if __name__ == '__main__':
    main()
