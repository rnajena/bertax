import argparse
from utils import (seq2tokens, get_token_dict, process_bert_tokens_batch, load_bert,
                   parse_fasta, annotate_predictions, best_predictions, seq_frames)
from os.path import splitext
from logging import warning, info, getLogger, INFO, WARNING
import numpy as np
from random import sample

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('fasta')
    parser.add_argument('--model_file', default='resources/big_trainingset_all_fix_classes_selection.h5',
                        help='path to the trained model to use')
    parser.add_argument('--conf_matrix', action='store_true')
    parser.add_argument('--long_sequence_behavior',
                        choices=['equal_chunks', 'all_frames', 'window'], default='equal_chunks')
    parser.add_argument('--long_sequence_aggregation',
                        choices=['mean', 'sum'], default='mean')
    parser.add_argument('--maximum_sequence_chunks', default=100, type=int,
                        help='maximum number of chunks to use for long sequences')
    parser.add_argument('--output_fmt', choices=['best_classes', 'all_values'],
                        default='best_classes')
    parser.add_argument('--output_ranks', help='rank predictions to include in output',
                        default=['superkingdom', 'phylum', 'genus'], nargs='+')
    parser.add_argument('--output_file', help='write output to specified file (Default: stdout)',
                        default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments()
    getLogger().setLevel(INFO if args.verbose else WARNING)
    out_prefix = splitext(args.fasta)[0]
    model = load_bert(args.model_file)
    max_seq_len = model.input_shape[0][1]
    token_dict = get_token_dict()
    # read input
    records = parse_fasta(args.fasta)
    # convert input to processable tokens
    out = []
    for record in records[:3]:
        if (np.ceil(len(record.seq) / 3) <= max_seq_len
            or args.long_sequence_behavior == 'window'):
            inputs = [seq2tokens(record.seq, token_dict, max_seq_len)]
        else:
            chunks = seq_frames(record.seq, max_seq_len * 3,
                                (args.long_sequence_behavior == 'all_frames'))
            if (len(chunks) > args.maximum_sequence_chunks):
                chunks = sample(chunks, args.maximum_sequence_chunks)
            inputs = [seq2tokens(chunk, token_dict, max_seq_len)
                      for chunk in chunks]
        info(f'converted sequence {record.id} (length {len(record.seq)}) into {len(inputs)} chunks')
        x = process_bert_tokens_batch(inputs)
        info(f'predicting sequence chunks for {record.id}')
        preds = model.predict(x, verbose=int(args.verbose))
        preds = list(map(lambda p: p.mean(axis=0), preds)) # TODO: args.long_sequence_aggregation
        annotated = annotate_predictions(preds)
        out.append((record.id, (best_predictions(annotated) if args.output_fmt == 'best_classes'
                                else annotated)))
    info(f'predicted classes for {len(records)} sequence records')
    if (args.output_fmt == 'best_classes'):
        field_len_thr = 50 # prevent overlong IDs
        out_values = [['id'] + args.output_ranks] + [
            list(map(lambda x: (x[:field_len_thr] + '...' if len(x) > field_len_thr
                           else x),
                     [o[0], *o[1]])) for o in out]
        max_field_lens = [max(o[i] for o in [list(map(len, o)) for o in out_values])
                          for i in range(len(out_values[0]))]
        row_format = ''.join(f'{{:<{f + 2}}}' for f in max_field_lens)
        print('\n'.join(row_format.format(*l) for l in out_values),
              file=(open(args.output_file, 'w') if args.output_file is not None else None))
