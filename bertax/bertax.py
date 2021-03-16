import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse
from bertax.utils import (seq2tokens, get_token_dict, process_bert_tokens_batch, load_bert,
                          parse_fasta, annotate_predictions, best_predictions, seq_frames)
import tensorflow as tf
from logging import  info, getLogger, INFO, WARNING
import numpy as np
from random import sample
import json
import pkg_resources

MAX_SIZE = 1500
FIELD_LEN_THR = 50 # prevent overlong IDs

def check_max_len_arg(value):
    value = int(value)
    if (value < 1 or value > MAX_SIZE):
        raise argparse.ArgumentTypeError(f'value has to be between 1 and {MAX_SIZE}')
    return value

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(prog='bertax',
                                     description='BERTax: Predicting sequence taxonomy',
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=42))
    parser.add_argument('fasta')
    parser.add_argument('-v', '--verbose', help='more verbose output if set, including progress bars',
                        action='store_true')
    parser.add_argument('-o', '--output_file', help='write output to specified file (Default: stdout)',
                        default=None, metavar='FILE')
    parser.add_argument('--conf_matrix_file', help='if set, writes confidences for all possible classes '
                        'in all ranks to specified file (JSON)', default=None, metavar='FILE')
    parser.add_argument('--sequence_split', help='how to handle sequences longer '
                        'than the maximum (window) size: split to equal chunks or '
                        'use random sequence window; also see `--chunk_predictions` '
                        'and `--running_window` (Default: equal_chunks)', choices=['equal_chunks', 'window'],
                        default='equal_chunks')
    parser.add_argument('--chunk_predictions', help='if set, predictions on chunks for long sequences '
                        'are not averaged (default) but printed individually in the output',
                        action='store_true')
    parser.add_argument('--running_window', action='store_true', help='if enabled, a running window '
                        'approach is chosen to go over each sequence with a default stride of `1` '
                        'and make predictions on those windows; takes priority over `sequence_split`')
    parser.add_argument('--running_window_stride', type=int, default=90, metavar='STRIDE',
                        help='stride (nt) for the running window approach (Default: 90)')
    parser.add_argument('-s', '--custom_window_size', default=None,
                        help='set custom, smaller window size (in nt, preferably multiples of 3), if unset, '
                        f'the maximum size of {MAX_SIZE} is used', metavar='SIZE', type=check_max_len_arg)
    parser.add_argument('-C', '--maximum_sequence_chunks', default=100, type=int,
                        help='maximum number of chunks to use for long sequences, '
                        '-1 to use all chunks (Default: 100)', metavar='NR')
    parser.add_argument('--output_ranks', help='rank predictions to include in output '
                        '(Default: superkingdom phylum genus)',
                        default=['superkingdom', 'phylum', 'genus'], nargs='+', metavar='RANK')
    parser.add_argument('--no_confidence', action='store_true',
                        help='do not include class confidence values in output')
    parser.add_argument('--batch_size', type=int, help='batch size for predictions (Default: 32)',
                        default=32)
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    getLogger().setLevel(INFO if args.verbose else WARNING)
    model_file = pkg_resources.resource_filename(
        'bertax', 'resources/big_trainingset_all_fix_classes_selection.h5')
    model = load_bert(model_file)
    max_seq_len = MAX_SIZE if args.custom_window_size is None else args.custom_window_size
    token_dict = get_token_dict()
    # read input
    records = parse_fasta(args.fasta)
    # convert input to processable tokens
    out = []
    for record in records:
        no_chunks = (not args.running_window and (len(record.seq) <= max_seq_len
                                                  or args.sequence_split == 'window'))
        if (no_chunks):
            inputs = [seq2tokens(record.seq, token_dict, np.ceil(max_seq_len / 3).astype(int),
                                 max_length=model.input_shape[0][1])]
        else:
            chunks, positions = seq_frames(record.seq, max_seq_len, args.running_window,
                                           args.running_window_stride)
            if (args.maximum_sequence_chunks > 0 and len(chunks) > args.maximum_sequence_chunks):
                info(f'sampling {args.maximum_sequence_chunks} from {len(chunks)} chunks')
                chunks, positions = list(zip(*sample(list(zip(chunks, positions)),
                                                     args.maximum_sequence_chunks)))
            inputs = [seq2tokens(chunk, token_dict, np.ceil(max_seq_len / 3).astype(int),
                                 max_length=model.input_shape[0][1]) for chunk in chunks]
        info(f'converted sequence {record.id} (length {len(record.seq)}) into {len(inputs)} chunks')
        x = process_bert_tokens_batch(inputs)
        info(f'predicting sequence chunks for {record.id}')
        preds = model.predict(x, verbose=int(args.verbose), batch_size=args.batch_size)
        if (args.chunk_predictions and not no_chunks):
            for pos, pred in zip(positions, [[p[i] for p in preds] for i in range(len(positions))]):
                out.append((f'[{pos[0]}..{pos[1]}] {record.id}', annotate_predictions(pred)))
        else:                   # for each window
            preds = list(map(lambda p: p.mean(axis=0), preds))
            annotated = annotate_predictions(preds)
            out.append((record.id, annotated))
    info(f'predicted classes for {len(records)} sequence records')
    ## OUTPUT
    if (args.output_file is None): # formatted table
        out_values = [['id'] + args.output_ranks] + [
            list(map(lambda x: (x[:FIELD_LEN_THR] + '...' if len(x) > FIELD_LEN_THR
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
