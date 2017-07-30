import os
import random
import sys

from optparse import OptionParser

from ehost2ann import read_project
from ann2bio import convert_to_bio_sequence, print_bio_sequence


def ehost_to_bio_sequence(project_name):
    bio_sequences = []
    for (doc_name, document) in read_project(project_name):
        bio_sequence = convert_to_bio_sequence(document)
        bio_sequences.append((doc_name, bio_sequence))
    return bio_sequences


def ehost2bio(project_name, filename, shuffle=False, delimiter='\t'):
    sequences = ehost2seq(project_name, shuffle)
    seq2bio(sequences, filename, delimiter)


def ehost2seq(project_name, shuffle):
    seqs = ehost_to_bio_sequence(project_name)
    file_names, seqs_per_file = zip(*seqs)
    sequences = [seq for sub_seqs in seqs_per_file for seq in sub_seqs]
    if shuffle:
        random.shuffle(sequences)
    return sequences


def seq2bio(bio_sequences, filename, delimiter):
    dirname, _ = os.path.split(filename)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with open(filename, "w") as out_file:
        print_bio_sequence(bio_sequences, file=out_file, delimiter=delimiter)


def ehost2bio_split(project_name, dirname, ratio, shuffle=False,
                    delimiter='\t', train_file="train.bio", test_file="test.bio"):
    sequences = ehost2seq(project_name, shuffle)
    seq2bio_split(sequences, dirname, ratio, delimiter, train_file, test_file)


def seq2bio_split(sequences, dirname, ratio, delimiter, train_file, test_file):
    total = len(sequences)
    num_train = int(ratio * total)
    train_sequences = sequences[:num_train]
    test_sequences = sequences[num_train:]
    train_file = os.path.join(dirname, train_file)
    test_file = os.path.join(dirname, test_file)
    seq2bio(train_sequences, train_file, delimiter)
    seq2bio(test_sequences, test_file, delimiter)


def main():
    parser = OptionParser(usage="usage: %prog [option] eHOST_project_dir output_file\n"
                                "\n"
                                "convert eHOST annotation project to IOB file"
                                "",
                          version="%prog 1.0")
    parser.add_option('-r', '--ratio', dest='ratio', help='percentage of training samples, if specified a folder will'
                                                          ' be created containing the training and test samples',
                      type=float)
    parser.add_option('-s', '--shuffle', action='store_true', dest='shuffle', help="shuffle the sequence")

    parser.add_option('-d', '--delimiter', dest='delimiter', help="delimiter to join "
                                                                  "the columns in BIO file, S=space, T=tab.")

    (options, args) = parser.parse_args()

    single_mode = True
    train_ratio = 1.0
    if options.ratio is not None:
        try:
            train_ratio = float(options.ratio)
            single_mode = False
        except ValueError:
            single_mode = True

    if options.delimiter is None:
        options.delimiter = '\t'
    elif options.delimiter == 'S':
        options.delimiter = ' '
    elif options.delimiter == 'T':
        options.delimiter = '\t'
    if len(args) < 2:
        parser.print_help()
        sys.exit(1)

    project_name, output_filename = args[:2]

    if single_mode:
        ehost2bio(project_name, output_filename, options.shuffle, options.delimiter)
    else:
        ehost2bio_split(project_name, output_filename, train_ratio, options.shuffle, options.delimiter)


if __name__ == "__main__":
    main()
