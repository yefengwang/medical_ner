# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jul 12 17:47:24 2017

@author: Yefeng Wang
"""

import sys
import json

from optparse import OptionParser

import pycrfsuite


class CharMap:
    """
    Mapping between Chinese characters and numbers.
    Chinese characters is not supported in the model.
    """
    def __init__(self):
        self.encoding_map = {}
        self.decoding_map = {}

    def __getitem__(self, char):
        index = self.encoding_map.get(char, len(self.encoding_map) + 1)
        self.encoding_map[char] = index
        self.decoding_map[str(index)] = char
        return str(self.encoding_map[char])

    def decode(self, idx_seq):
        return ''.join([self.decoding_map.get(idx, 'UKN') for idx in idx_seq.split('-')])

    def encode(self, word):
        chars = toknise_chinese(word)
        char_idxs = [self.__getitem__(char) for char in chars]
        word_idxs = "-".join(char_idxs)
        return word_idxs

    def encode_label(self, label):
        p, l = split_label(label)
        idx = self.encode(l)
        return p + idx

    def encode_labels(self, labels):
        return [self.encode_label(label) for label in labels]

    def decode_labels(self, labels):
        return [self.decode_label(label) for label in labels]

    def decode_label(self, label):
        p, l = split_label(label)
        word = self.decode(l)
        return p + word

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(json.dumps(self.encoding_map, indent=4, sort_keys=True))

    def load(self, filename):
        with open(filename, "r") as f:
            self.encoding_map = json.load(f)
            for (word, idx) in self.encoding_map.items():
                self.decoding_map[str(idx)] = word


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def split_label(label):
    if label[0] in ['B', 'I']:
        return label[:2], label[2:]
    else:
        return '', label


def toknise_chinese(word):
    if is_ascii(word):
        return [word]
    else:
        return list(word)


def load_bio_file(filename):
    with open(filename, "r") as inf:
        return load_bio_content(inf.read())


def load_bio_content(output_file_content):
    """
    Load the BIO files, return sentences as a list of (word, tag)
    """

    sentences = []
    sentence = []
    for line in output_file_content.split("\n"):
        if not line.strip():
            if sentence:
                sentences.append(sentence)
                sentence = []
            continue
        row = line.split()
        word = row[0]
        pos = row[1]
        label = row[-1]
        sentence.append((word, pos, label))

    return sentences


def word2features(sent, i, char_map):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word=' + char_map.encode(word),
        'postag=' + postag,
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word=' + char_map.encode(word1),
            '-1:postag=' + postag1,
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word=' + char_map.encode(word1),
            '+1:postag=' + postag1,
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent, char_map):
    return [word2features(sent, i, char_map) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


class Learner:

    def __init__(self, char_map=None):
        if char_map is None:
            self.char_cache = CharMap()
        else:
            self.char_cache = char_map

    def train(self, train_filename, model_name):
        train_sents = load_bio_file(train_filename)

        x_train = [sent2features(s, self.char_cache) for s in train_sents]
        y_train = [self.char_cache.encode_labels(sent2labels(s)) for s in train_sents]

        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(x_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 200,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        model_filename = model_name + ".model"
        charmap_filename = model_name + ".charmap"

        trainer.train(model_filename)

        self.char_cache.save(charmap_filename)

        print("done.")


class Classifier:

    def __init__(self, model_filename, char_map_filename, char_map=None):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(model_filename)
        if char_map is None:
            self.char_cache = CharMap()
            self.char_cache.load(char_map_filename)
        else:
            self.char_cache = char_map

    def predict_file(self, test_filename):

        test_sents = load_bio_file(test_filename)
        x = [sent2features(s, self.char_cache) for s in test_sents]
        y = [self.char_cache.decode_labels(y_) for y_ in self.tagger.tag(x)]
        return y

    def predict(self, sentence):
        x = sent2features(sentence, self.char_cache)
        y = self.char_cache.decode_labels(self.tagger.tag(x))
        return y


def train(input_filename, model_name):
    learner = Learner()
    learner.train(input_filename, model_name)


def predict(input_filename, model_name):

    model_filename = model_name + ".model"
    charmap_filename = model_name + ".charmap"
    classifier = Classifier(model_filename, charmap_filename)

    test_sents = load_bio_file(input_filename)

    for sentence in test_sents:
        tokens = sent2tokens(sentence)
        predict = classifier.predict(sentence)
        correct = sent2labels(sentence)
        for (token, corr, pred) in zip(tokens, correct, predict):
            print("%s %s %s" % (token, corr, pred))
        print()


def main():
    parser = OptionParser(usage="usage: %prog [option] input_file output_file\n"
                                "\n"
                                "train or make prediction for Chinese Medical Named Entity Recognition"
                                "",
                          version="%prog 1.0")
    parser.add_option('-t', '--train', action='store_true', dest='train_model', help='train a crf model')
    parser.add_option('-m', '--model', dest='model_file', help="model file")

    (options, args) = parser.parse_args()

    if options.model_file is None or not options.model_file.strip():
        parser.error("Missing model filename")
        sys.exit(1)

    if len(args) < 1:
        parser.error("Incorrect number of arguments")
        sys.exit(1)

    model_name = options.model_file.strip()
    input_filename = args[0]

    if options.train_model:
        train(input_filename, model_name)
    else:
        predict(input_filename, model_name)

if __name__ == "__main__":
    main()
