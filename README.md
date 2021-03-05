# BERTax: Taxonomic Classification of DNA sequences

## Installation

Simply install with pip:
```shell
pip install bertax
```

Optionally, create clean virtual environment beforehand:
```shell
# venv
python -m venv bertax
source bertax/bin/activate

# conda
conda create -n bertax
conda activate bertax
```

## Usage

The script takes a (multi)fasta as input and outputs a list of predicted classes:

```shell
bertax sequences.fasta
```

Options:
| parameter                    | explanation                                                                                                                                                                         |
| ---                          | ---                                                                                                                                                                                 |
| -o --output_file             | write output to specified file (tab-separated format) instead of to the output stream (console)                                                                                     |
| --conf_matrix_file           | output confidences for all classes of all ranks to JSON file                                                                                                                        |
| --long_sequence_behavior     | how to handle long sequences (longer than 1500nt): split into equal chunks (`equal_chunks`, default), use all possible frames (`all_frames`), use 1500nt sequence window (`window`) |
| -C --maximum_sequence_chunks | maximum number of chunks to use per (long) sequence                                                                                                                                 |
| --output_ranks               | specify which ranks to include in output (default: superkingdom phylum genus)                                                                                                       |
| --no_confidence              | if set, do not include confidence scores in output                                                                                                                                  |
| --batch_size                 | batch size (i.e., how many sequence chunks to predict at the same time); can be lowered to decrease memory usage and increased for better performance (default: 32)                 |

Note, that "unknown" is a special placeholder class for each prediction rank, meaning the sequence's taxonomy is predicted to be unlike any possible output class.
