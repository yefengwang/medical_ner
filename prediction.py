import os
import sys
import tensorflow as tf

from preprocessing import prepare_document, prepare_sentence

from NERModel import Vocab, Model
from NERModel import get_vocab_filenames

class NERClassifier:

    def __init__(self, input_dir):
        model_dir = os.path.join(input_dir, "model")

        word_vocab_filename, char_vocab_filename, label_vocab_filename = get_vocab_filenames(input_dir)

        self.char_vocab = Vocab(char_vocab_filename, encode_char=True)
        self.word_vocab = Vocab(word_vocab_filename)
        self.label_vocab = Vocab(label_vocab_filename, encode_tag=True)

        num_words = len(self.word_vocab)
        num_chars = len(self.char_vocab)
        num_labels = len(self.label_vocab)

        # max_sentence_len = train.max_length()
        max_sentence_len = 100

        max_word_len = 20

        model = Model(num_words, num_labels, num_chars, max_sentence_len, max_word_len, model_dir, load_model=True)
        self.model = model

    def predict(self, input_seq):
        return self.model.predict(input_seq, self.word_vocab, self.char_vocab, self.label_vocab)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        sys.exit(0)

    classifier = NERClassifier(sys.argv[1])

    document = open(sys.argv[2]).read()
    sentences = prepare_document(document)
    for sentence in sentences:
        tokens, position, pos, tags = zip(*sentence)
        tags = classifier.predict(tokens)
        for (token, tag) in zip(tokens, tags):
            print("{}\t{}".format(token, tag))
        print()
