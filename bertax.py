import argparse
from utils import (seq2tokens, get_token_dict, process_bert_tokens_batch, load_bert,
                   parse_fasta, annotate_predictions, best_predictions, seq_frames)
from logging import  info, getLogger, INFO, WARNING
import numpy as np
from random import sample
import json

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='BERTax: Predicting sequence taxonomy')
    parser.add_argument('fasta')
    parser.add_argument('-v', '--verbose', help='more verbose output if set, including progress bars',
                        action='store_true')
    parser.add_argument('-o', '--output_file', help='write output to specified file (Default: stdout)',
                        default=None)
    parser.add_argument('--conf_matrix_file', help='if set, writes confidences for all possible classes in all ranks to specified file (JSON)',
                        default=None)
    parser.add_argument('--long_sequence_behavior', help='how to handle sequences longer than the maximum size: '
                        'split to equal chunks, use all possible sequence frames, use random sequence window; '
                        'for multiple chunks, the mean is calculated (Default: equal_chunks)',
                        choices=['equal_chunks', 'all_frames', 'window'], default='equal_chunks')
    parser.add_argument('-C', '--maximum_sequence_chunks', default=10, type=int,
                        help='maximum number of chunks to use for long sequences')
    parser.add_argument('--output_ranks', help='rank predictions to include in output '
                        '(Default: superkingdom phylum genus)',
                        default=['superkingdom', 'phylum', 'genus'], nargs='+')
    parser.add_argument('--no_confidence', action='store_true',
                        help='do not include class confidence values in output')
    parser.add_argument('--batch_size', type=int, help='batch size for predictions (Default: 32)',
                        default=32)
    parser.add_argument('--model_file', default='resources/big_trainingset_all_fix_classes_selection.h5',
                        help='path of the trained model to use (Default: "resources/big_trainingset_all_fix_classes_selection.h5")')
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    getLogger().setLevel(INFO if args.verbose else WARNING)
    model = load_bert(args.model_file)
    max_seq_len = model.input_shape[0][1]
    token_dict = get_token_dict()
    # read input
    records = parse_fasta(args.fasta)
    # convert input to processable tokens
    out = []
    for record in records:
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
        preds = model.predict(x, verbose=int(args.verbose), batch_size=args.batch_size)
        preds = list(map(lambda p: p.mean(axis=0), preds))
        annotated = annotate_predictions(preds)
        out.append((record.id, annotated))
    info(f'predicted classes for {len(records)} sequence records')
    ## OUTPUT
    if (args.output_file is None): # formatted table
        field_len_thr = 50 # prevent overlong IDs
        out_values = [['id'] + args.output_ranks] + [
            list(map(lambda x: (x[:field_len_thr] + '...' if len(x) > field_len_thr
                           else x),
                     [o[0],     # id
                      *[f'{p[0]} ({p[1]:.0%})' if not args.no_confidence else p[0]
                        for p in best_predictions(o[1])] # best prediction per rank
                      ])) for o in out]
        max_field_lens = [max(o[i] for o in [list(map(len, o)) for o in out_values])
                          for i in range(len(out_values[0]))]
        row_format = ''.join(f'{{:<{f + 2}}}' for f in max_field_lens)
        print('\n'.join(row_format.format(*l) for l in out_values))
    else:                       # tsv
        with open(args.output_file, 'w') as handle:
            handle.write('\t'.join(['id'] + [f'{rank}\t(%)' for rank in args.output_ranks]) + '\n')
            for id_, annots in out:
                handle.write('\t'.join([id_] + [f'{b[0]}\t{b[1]:.0%}' for b in best_predictions(annots)]) + '\n')
    if (args.conf_matrix_file is not None):
        json.dump(out, open(args.conf_matrix_file, 'w'), indent=2)
if __name__ == '__main__':
    main()
