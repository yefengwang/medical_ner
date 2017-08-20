"""
A trainable CNN+LSTM+CRF Named Entity Recogniser

@author Yefeng Wang
@date 2017-07-28
@version 1.0
"""

import numpy as np
import os
import tensorflow as tf
import sys

from optparse import OptionParser

"""
Epoch 35 out of 35
..............................
- dev acc 84.54 - f1 64.99
- new best score!
Testing model over test set
- test acc 87.94 - f1 74.78

# Hyper parameters
word_embedding_size = 500 # word embedding size
char_embedding_size = 300 # char embedding size
filter_sizes = [2] # CNN filter sizes, use window 3, 4, 5 for char CNN
num_filters = 200 # CNN output
hidden_size = 300 # LSTM hidden size

converge_check = 20
use_chars = True
use_crf = True
clip = 5
batch_size = 20
num_epochs = 35
dropout = 0.5
learning_rate = 0.001
learning_rate_decay = 0.9
"""



# Hyper parameters
word_embedding_size = 500 # word embedding size
char_embedding_size = 300 # char embedding size
filter_sizes = [2] # CNN filter sizes, use window 3, 4, 5 for char CNN
num_filters = 200 # CNN output
hidden_size = 300 # LSTM hidden size

converge_check = 20
use_chars = True
use_crf = True
clip = 5
batch_size = 20
num_epochs = 35
dropout = 0.5
learning_rate = 0.001
learning_rate_decay = 0.9

reload_model = False

UNK = "<<UNK>>"
NUM = "<<NUM>>"
OUTSIDE = "O"


def get_vocabs(datasets):
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    return vocab_words, vocab_tags


class BIOFileLoader(object):
    def __init__(self, filename, word_vocab=None, char_vocab=None, tag_vocab=None):
        self.filename = filename
        #self.max_seq = max_seq
        self._length = None
        self._max_length = None
        self._max_word_length = None
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab

    def _initialize(self):
        length, max_length, max_word_length = 0, 0, 0
        for (xws, xcs), ys in self:
            length += 1
            max_length = max(len(xws), max_length)
            max_word_length = max(max_word_length, max(map(len, xcs)))
            print(max_length, max_word_length)
        self._max_length = max_length
        self._max_word_length = max_word_length
        self._length = length

    def max_length(self):
        if self._max_length is None:
            self._initialize()

        return self._max_length

    def max_word_length(self):
        if self._max_word_length is None:
            self._initialize()

        return self._max_word_length

    def __iter__(self):
        num_seq = 0
        with open(self.filename, "r") as f:
            words, chars, tags = [], [], []
            for line in f:
                line = line.strip()
                if not line or line.startswith("-DOCSTART-"):
                    if words:
                        num_seq += 1
                        yield (words, chars), tags
                        words, chars, tags = [], [], []
                else:
                    ls = line.split(' ')
                    word, char, tag = ls[0], list(ls[0]), ls[-1]
                    if self.word_vocab is not None:
                        word = self.word_vocab.encode(word)
                    if self.char_vocab is not None:
                        char = self.char_vocab.encode(char)
                    if self.tag_vocab is not None:
                        tag = self.tag_vocab.encode(tag)
                    words.append(word)
                    chars.append(char)
                    tags.append(tag)

    def __len__(self):
        if self._length is None:
            self._initialize()

        return self._length


class Vocab:

    def __init__(self, filename=None, encode_char=False, encode_tag=False):
        self.max_idx = 0
        self.encoding_map = {}
        self.decoding_map = {}
        self.encode_char = encode_char
        self.encode_tag = encode_tag
        self._insert = True
        if not self.encode_tag:
            self._encode(UNK, add=True)
            if not self.encode_char:
                self._encode(NUM, add=True)
        else:
            self._encode(OUTSIDE, add=True)
        if filename:
            self.load(filename)
            self._insert = False
        # add the special char to the set anyway.
        if not self.encode_tag:
            self._encode(UNK, add=True)
            if not self.encode_char:
                self._encode(NUM, add=True)
        else:
            self._encode(OUTSIDE, add=True)

    def __len__(self):
        #return len(self.encoding_map)
        return self.max_idx + 1

    def encodes(self, seq):
        '''
        encode a sequence
        '''
        return [self.encode(word) for word in seq]

    def encode(self, word):
        '''
        encode a word or a char
        '''
        if self.encode_char:
            return [self._encode(char, add=self._insert) for char in word]
        else:
            return self._encode(word, add=self._insert)

    def encode_datasets(self, datasets):
        for dataset in datasets:
            for (xws, xcs), ys in dataset:
                if self.encode_tag:
                    self.encodes(ys)
                elif self.encode_char:
                    self.encodes(xcs)
                else:
                    self.encodes(xws)

    def decodes(self, idxs):
        return map(self.decode, idxs)

    def decode(self, idx):
        return self.decoding_map.get(idx, UNK)

    def _encode(self, word, lower=False, add=False):
        if lower:
            word = word.lower()
        if add:
            idx = self.encoding_map.get(word, self.max_idx + 1)
            self.max_idx = max(idx, self.max_idx)
            self.encoding_map[word] = idx
            self.decoding_map[idx] = word
        else:
            if self.encode_tag:
                idx = self.encoding_map.get(word, self.encoding_map[OUTSIDE])
            else:
                idx = self.encoding_map.get(word, self.encoding_map[UNK])

        return idx

    def save(self, filename):
        with open(filename, "w") as f:
            for word, idx in self.encoding_map.items():
                f.write("%s\t%s\n" % (word, idx))

    def load(self, filename):
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                word, idx = line.split("\t")
                idx = int(idx)
                self.encoding_map[word] = idx
                self.decoding_map[idx] = word
                self.max_idx = max(idx, self.max_idx)


class Model(object):

    def __init__(self, num_words, num_tags, num_chars, max_sentence_len, 
                 max_word_len, learning_rate, model_dir, word_embeddings=None):
        self.num_words = num_words
        self.num_chars = num_chars
        self.num_tags = num_tags
        self.max_word_len = max_word_len
        self.max_sentence_len = max_sentence_len
        self.learning_rate = learning_rate
        self.model_dir = model_dir


        # shape = (batch size, max length of sentence in batch)
        self.words = tf.placeholder(tf.int32, shape=[None, None], name="words")
        # shape = (batch size, max length of sentence, max length of word)
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name="chars")
        # shape = (batch size, max length of sentence in batch)
        self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")
        # shape = (batch size)
        self.sentences_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")
        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.train_op = None
        self.loss = None

        with tf.variable_scope("words"):
            if word_embeddings is not None:
                word_embeddings_W = tf.Variable(word_embeddings, name="word_embedding_w", dtype=tf.float32, trainable=True)
                word_embeddings = tf.nn.embedding_lookup(word_embeddings_W, self.words, name="word_embeddings")
            else:
                word_embeddings_W = tf.get_variable(name="word_embeddings_w", dtype=tf.float32,
                                                    shape=[self.num_words, word_embedding_size],
                                                    initializer=tf.random_normal_initializer())
                word_embeddings = tf.nn.embedding_lookup(word_embeddings_W, self.words, name="word_embeddings")

        with tf.variable_scope("chars"):
            if use_chars:
                char_embeddings_W = tf.get_variable(name="char_embeddings_w", dtype=tf.float32,
                                                    shape=[self.num_chars, char_embedding_size],
                                                    initializer=tf.random_normal_initializer())

                char_embeddings = tf.nn.embedding_lookup(char_embeddings_W, self.chars, name="char_embeddings")
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                #s = tf.Print(s, [s], "s=", summarize=10)

                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], char_embedding_size])
                #char_embeddings = tf.Print(char_embeddings, [tf.shape(char_embeddings)], "embedding_shape=",
                #                           summarize=10)
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])
                #word_lengths = tf.Print(word_lengths, [word_lengths], "word_lengths=", summarize=10)
                """
                # word level LSTM
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, 
                                                    state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size, 
                                                    state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                    cell_bw, char_embeddings, sequence_length=word_lengths, 
                    dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)
                output = tf.Print(output, [tf.shape(output)], "output=", summarize=10)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, s[1], 2*self.config.char_hidden_size])
                """

                pooled_outputs = []
                # add channel
                expanded_char_embeddings = tf.expand_dims(char_embeddings, -1)
                for i, filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, char_embedding_size, 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                        conv = tf.nn.conv2d(
                            expanded_char_embeddings,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # Maxpooling over the outputs, only on height
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.max_word_len - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = num_filters * len(filter_sizes)
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, s[1], num_filters_total])
                #h_pool_flat = tf.Print(h_pool_flat, [tf.shape(h_pool_flat)], "pool=", summarize=10)
                word_embeddings = tf.concat([word_embeddings, h_pool_flat], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw,
                                                                        self.word_embeddings,
                                                                        sequence_length=self.sentences_lengths,
                                                                        dtype=tf.float32)
            lstm_output = tf.concat([output_fw, output_bw], axis=-1)
            lstm_output = tf.nn.dropout(lstm_output, self.dropout)

        self.logits = None
        with tf.variable_scope("fc"):
            W = tf.get_variable("W",
                                shape=[2 * hidden_size, self.num_tags],
                                dtype=tf.float32)

            b = tf.get_variable("b",
                                shape=[self.num_tags],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            num_time_steps = tf.shape(lstm_output)[1]
            lstm_output = tf.reshape(lstm_output, [-1, 2 * hidden_size])
            pred = tf.matmul(lstm_output, W) + b
            self.logits = tf.reshape(pred, [-1, num_time_steps, self.num_tags])

        if not use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        if use_crf:
            #self.sentences_lengths = tf.Print(self.sentences_lengths, [self.sentences_lengths], "s=", summarize=10)
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.tags, self.sentences_lengths)

            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tags)
            mask = tf.sequence_mask(self.sentences_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # train process
        with tf.variable_scope("train_step"):
            optimizer = tf.train.RMSPropOptimizer(self.lr)

            self.train_op = None
            # gradient clipping if config.clip is positive
            if clip > 0:
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, global_norm = tf.clip_by_global_norm(gradients, clip)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def train(self, train, dev, tags):
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if reload_model:
                print("Reloading the latest trained model...")
                saver.restore(sess, self.model_dir)
            for epoch in range(num_epochs):
                print("Epoch {:} out of {:}".format(epoch + 1, num_epochs))

                # num_instance = len(train)
                train_loss = 0.0
                for i, (words, labels) in enumerate(mini_batch(train, batch_size)):
                    fd, _ = self.get_feed_dict(words, labels, learning_rate, dropout)
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    _, train_loss = sess.run([self.train_op, self.loss], feed_dict=fd)
                print()
                acc, f1 = self.evaluate(sess, dev, tags)
                print("# loss {:04.8f} acc {:04.2f} f1 {:04.2f}".format(train_loss, 100*acc, 100*f1))

                # decay learning rate
                self.learning_rate  *= learning_rate_decay
                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.model_dir):
                        os.makedirs(self.model_dir)
                    saver.save(sess, os.path.join(self.model_dir, "model"))
                    best_score = f1
                    print("# best model")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= converge_check:
                        print("# stopped after {} epochs without improvement".format(
                            nepoch_no_imprv))
                        break


    def predict_batch(self, sess, words):
        feed_dict, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if use_crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
                viterbi_sequences += [viterbi_sequence]
            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=feed_dict)

            return labels_pred, sequence_lengths

    def test(self, test, tags):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Testing model over test set")
            saver.restore(sess, os.path.join(self.model_dir, "model"))
            acc, f1 = self.evaluate(sess, test, tags)
            print("# test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))

    def evaluate(self, sess, test, tags):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in mini_batch(test, batch_size):
            labels_pred, sequence_lengths = self.predict_batch(sess, words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, tags))
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)
        return acc, f1


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        words, chars = zip(*words)

        words, sentences_lengths = pad_sentence(words, 0, max_length=self.max_sentence_len)
        chars, word_lengths = pad_chars(chars, pad_tok=0,
                                        max_word_length=self.max_word_len, max_sentence_length=self.max_sentence_len)

        #print(chars)
        #import sys
        #sys.exit(0)
        feed = {
            self.words: words,
            self.sentences_lengths: sentences_lengths,
            self.chars: chars,
            self.word_lengths: word_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels, 0, self.max_sentence_len)
            feed[self.tags] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sentences_lengths


def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    default = tags[OUTSIDE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def mini_batch(data, batch_size):
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch += [map(list, x)]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def pad_sequences(sequences, pad_tok, max_length):

    padded_sequences = []
    sequence_length = []

    for seq in sequences:
        seq = list(seq)
        padded_seq = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        padded_sequences.append(padded_seq)
        sequence_length += [min(len(seq), max_length)]

    return padded_sequences, sequence_length


def pad_sentence(sentences, pad_tok, max_length=-1):
    if max_length == -1:
        max_length = max(map(lambda x : len(x), sentences))
    padded_sentences, sentence_length = pad_sequences(sentences, pad_tok, max_length)
    return padded_sentences, sentence_length


def pad_chars(sentences, pad_tok, max_word_length=-1, max_sentence_length=-1):
    if max_word_length <= 0:
        max_word_length = max([max(map(lambda x: len(x), seq)) for seq in sentences])
    padded_words = []
    word_length = []
    for seq in sentences:
        # all words are same length now
        word, word_len = pad_sequences(seq, pad_tok, max_word_length)
        padded_words.append(word)
        word_length.append(word_len)

    if max_sentence_length <= 0:
        max_sentence_length = max(map(lambda x : len(x), sentences))
    # pad the sentence with empty words of length max_word_length
    padded_sentences, _ = pad_sequences(padded_words, [pad_tok]*max_word_length,
                                        max_sentence_length)
    # pad the word length array as well
    sentence_length, _ = pad_sequences(word_length, 0, max_sentence_length)

    return padded_sentences, sentence_length


def get_vocab_filenames(working_dir):
    char_vocab_filename = os.path.join(working_dir, "char_vocab.txt")
    word_vocab_filename = os.path.join(working_dir, "word_vocab.txt")
    label_vocab_filename = os.path.join(working_dir, "label_vocab.txt")
    return word_vocab_filename, char_vocab_filename, label_vocab_filename


def get_input_filenames(input_dir):
    train_filename = os.path.join(input_dir, "train.txt")
    valid_filename = os.path.join(input_dir, "valid.txt")
    test_filename = os.path.join(input_dir, "test.txt")
    return train_filename, valid_filename, test_filename

def main():
    parser = OptionParser(usage="usage: %prog [option] data_dir\n"
                                "\n"
                                "build a cnn+lstm+crf named entity recogniser."
                                "",
                          version="%prog 1.0")
    parser.add_option('-b', '--build', action='store_true', dest='build', help='build the vocabulary')

    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.error("Invalid number of argument.")

    input_dir = args[0]

    if options.build:
        build(input_dir)
    else:
        run(input_dir)

def run(input_dir):

    model_dir = os.path.join(input_dir, "model")
    
    train_filename, valid_filename, test_filename = get_input_filenames(input_dir)
    word_vocab_filename, char_vocab_filename, label_vocab_filename = get_vocab_filenames(input_dir)

    char_vocab = Vocab(char_vocab_filename, encode_char=True)
    word_vocab = Vocab(word_vocab_filename)
    tag_vocab = Vocab(label_vocab_filename, encode_tag=True)

    train = BIOFileLoader(train_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)
    dev = BIOFileLoader(valid_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)
    test = BIOFileLoader(test_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)

    num_words = len(word_vocab)
    num_chars = len(char_vocab)
    num_labels = len(tag_vocab)

    max_sentence_len = train.max_length()

    max_word_len = min(train.max_word_length(), 20)

    print("max_sentence={}".format(max_sentence_len))
    print("max_word={}".format(max_word_len))

    model = Model(num_words, num_labels, num_chars, max_sentence_len, max_word_len, learning_rate, model_dir)

    vocab_tags = tag_vocab.encoding_map

    model.train(train, dev, vocab_tags)
    model.test(test, vocab_tags)

def build(input_dir):
    
    working_dir = input_dir 
    # os.makedirs(working_dir, exist_ok=True)
    train_filename, valid_filename, test_filename = get_input_filenames(input_dir)
    word_vocab_filename, char_vocab_filename, label_vocab_filename = get_vocab_filenames(working_dir)

    char_vocab = Vocab(encode_char=True)
    word_vocab = Vocab()
    label_vocab = Vocab(encode_tag=True)

    train = BIOFileLoader(train_filename)
    valid = BIOFileLoader(valid_filename)
    test = BIOFileLoader(test_filename)

    char_vocab.encode_datasets([train, valid])
    word_vocab.encode_datasets([train, valid])
    label_vocab.encode_datasets([train, valid])

    print("word vocab size {}".format(len(word_vocab)))
    print("char vocab size {}".format(len(char_vocab)))
    print("labal vocab size {}".format(len(label_vocab)))

    char_vocab.save(char_vocab_filename)
    word_vocab.save(word_vocab_filename)
    label_vocab.save(label_vocab_filename)


if __name__ == "__main__":
    main()

