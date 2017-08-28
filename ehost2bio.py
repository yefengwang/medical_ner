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
                    delimiter=' ', train_file="train.txt", test_file="test.txt"):
    sequences = ehost2seq(project_name, shuffle)
    seq2bio_split(sequences, dirname, ratio, delimiter, train_file, test_file)


def ehost2bio_create_fold(project_name, dirname, total_folds, shuffle=False, delimiter=' ',
                          train_file="train.txt", valid_file="valid.txt", test_file="test.txt"):
    sequences = ehost2seq(project_name, shuffle)
    for fold_num in range(total_folds):
        print("Creating fold %s ..." % fold_num)
        _ehost2bio_create_fold(sequences, dirname, fold_num, total_folds, delimiter,
                               train_file, valid_file, test_file)



def _ehost2bio_create_fold(sequences, dirname, fold_num, total_folds, delimiter=' ',
                     train_file="train.txt", valid_file="valid.txt", test_file="test.txt"):
    fold_dir = "fold_%s" % fold_num
    total = len(sequences)
    num_per_fold = total // total_folds
    test_start = ((total_folds + fold_num) % total_folds) * num_per_fold
    test_end = test_start + num_per_fold
    valid_start = ((total_folds + fold_num - 1) % total_folds) * num_per_fold
    valid_end = valid_start + num_per_fold
    print("valid:", valid_start, valid_end)
    print("test:", test_start, test_end)
    test_sequences = sequences[test_start:test_end]
    valid_sequences = sequences[valid_start:valid_end]
    if test_end < valid_start:
        train_sequences = sequences[test_end: valid_start]
    else:
        train_sequences = sequences[:valid_start] + sequences[test_end:]
    print("{}, {}, {}".format(len(train_sequences), len(valid_sequences), len(test_sequences)))
    train_file = os.path.join(dirname, fold_dir, train_file)
    valid_file = os.path.join(dirname, fold_dir, valid_file)
    test_file = os.path.join(dirname, fold_dir, test_file)
    seq2bio(train_sequences, train_file, delimiter)
    seq2bio(valid_sequences, valid_file, delimiter)
    seq2bio(test_sequences, test_file, delimiter)


def ehost2bio_create(project_name, dirname, train_ratio, valid_ratio, shuffle=False, delimiter=' ',
                     train_file="train.txt", valid_file="valid.txt", test_file="test.txt"):
    sequences = ehost2seq(project_name, shuffle)
    total = len(sequences)
    num_train = int(train_ratio * total)
    num_valid = int(valid_ratio * total)
    num_test = int(total - num_train - num_valid)
    print("{}, {}, {}".format(num_train, num_valid, num_test))
    train_sequences = sequences[:num_train]
    valid_sequences = sequences[num_train:num_train+num_valid]
    test_sequences = sequences[num_train+num_valid:]
    train_file = os.path.join(dirname, train_file)
    valid_file = os.path.join(dirname, valid_file)
    test_file = os.path.join(dirname, test_file)
    seq2bio(train_sequences, train_file, delimiter)
    seq2bio(valid_sequences, valid_file, delimiter)
    seq2bio(test_sequences, test_file, delimiter)


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

    parser.add_option('-f', '--fold', dest='num_folds', help="Create n fold cross validation data.", type=int)

    parser.add_option('-c', '--create', action='store_true', dest='create',
                      help="create a training, validation and test set.")
    (options, args) = parser.parse_args()

    total_folds = 0
    mode = 'normal'
    train_ratio = 1.0
    if options.ratio is not None:
        try:
            train_ratio = float(options.ratio)
            mode = "split"
        except ValueError:
            mode = "normal"
    if options.num_folds:
        mode = "fold"
        total_folds = int(options.num_folds)
    if options.create:
        mode = 'create'


    if options.delimiter is None:
        options.delimiter = ' '
    elif options.delimiter == 'S':
        options.delimiter = ' '
    elif options.delimiter == 'T':
        options.delimiter = '\t'
    if len(args) < 2:
        parser.print_help()
        sys.exit(1)

    project_name, output_filename = args[:2]

    if mode == 'normal':
        ehost2bio(project_name, output_filename, options.shuffle, options.delimiter)
    elif mode == 'fold':
        ehost2bio_create_fold(project_name, output_filename, total_folds, shuffle=options.shuffle, delimiter=' ',
                                  train_file="train.txt", valid_file="valid.txt", test_file="test.txt")
    elif mode == 'split':
        ehost2bio_split(project_name, output_filename, train_ratio, options.shuffle, options.delimiter)
    elif mode == 'create':
        train_ratio = 0.7
        valid_ratio = 0.15
        ehost2bio_create(project_name, output_filename, train_ratio, valid_ratio,
                         shuffle=options.shuffle, delimiter=options.delimiter)

if __name__ == "__main__":
    main()
