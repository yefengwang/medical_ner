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
import collections

from Vocab import Vocab
from Vocab import NUM, PAD, OUTSIDE, UNK

from save_embeddings import read_embeddings_vocab, save_char_embeddings, save_word_embeddings, load_embeddings

# Hyper parameters
word_embedding_size = 200 # word embedding size
char_embedding_size = 200 # char embedding size
kernels = [2, 3] # CNN filter sizes, use window 3, 4, 5 for char CNN
char_hidden_size = 200 # CNN output
lstm_hidden_size = 400 # LSTM hidden size

converge_check = 30
use_chars = True
char_embedding_method = "hcnn"
#char_embedding_method = "lstm"
#char_embedding_method = "vcnn"
use_crf = True
use_char_attention = True
clip = 5
batch_size = 20
num_epochs = 35
dropout = 0.5
learning_rate = 0.001
learning_rate_decay = 0.9

reload_model = False

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
            #print(max_length, max_word_length)
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


def viterbi_decode(score, transition_params, invalid_transitions):
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    # The invalid sequences should not produce any transition
    #for i, j in invalid_transitions:
    #    transition_params[i, j] = -100000.0

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


class Model(object):

    def __init__(self, num_words, num_tags, num_chars, max_sentence_len, 
                 max_word_len, model_dir,
                 word_embeddings=None, char_embeddings=None,
                 load_model=False, invalid_transitions=[]):
        tf.reset_default_graph()
        self.num_words = num_words
        self.num_chars = num_chars
        self.num_tags = num_tags
        self.max_word_len = max_word_len
        self.max_sentence_len = max_sentence_len
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.invalid_transitions = invalid_transitions

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

        self.transition_params = tf.get_variable(
            "transitions",
            shape=[self.num_tags, self.num_tags],
            initializer=tf.zeros_initializer())


        self.train_op = None
        self.loss = None
        with tf.variable_scope("words"):
            if word_embeddings is not None:
                with tf.device('/cpu:0'):
                    word_embeddings_W = tf.Variable(word_embeddings, name="word_embedding_w", dtype=tf.float32, trainable=True)
                    word_embeddings = tf.nn.embedding_lookup(word_embeddings_W, self.words, name="word_embeddings")
            else:
                word_embeddings_W = tf.get_variable(name="word_embeddings_w", dtype=tf.float32,
                                                    shape=[self.num_words, word_embedding_size],
                                                    initializer=tf.random_normal_initializer())
                word_embeddings = tf.nn.embedding_lookup(word_embeddings_W, self.words, name="word_embeddings")

        with tf.variable_scope("chars"):
            if use_chars:
                if char_embeddings is not None:
                    with tf.device('/cpu:0'):
                        char_embeddings_W = tf.Variable(char_embeddings, name="char_embeddings_w", dtype=tf.float32, trainable=True)
                        char_embeddings = tf.nn.embedding_lookup(char_embeddings_W, self.chars, name="char_embeddings")
                else:
                    char_embeddings_W = tf.get_variable(name="char_embeddings_w", dtype=tf.float32,
                                                        shape=[self.num_chars, char_embedding_size],
                                                        initializer=tf.random_normal_initializer())
                    char_embeddings = tf.nn.embedding_lookup(char_embeddings_W, self.chars, name="char_embeddings")

                s = tf.shape(char_embeddings)
                #s = tf.Print(s, [s], "s=", summarize=10)

                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], char_embedding_size])

                if char_embedding_method == "lstm":
                    word_lengths = tf.reshape(self.word_lengths, shape=[-1])

                    # word level LSTM
                    cell_fw = tf.contrib.rnn.LSTMCell(char_hidden_size,
                                                        state_is_tuple=True)
                    cell_bw = tf.contrib.rnn.LSTMCell(char_hidden_size,
                                                        state_is_tuple=True)

                    _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                        cell_bw, char_embeddings, sequence_length=word_lengths,
                        dtype=tf.float32)

                    output = tf.concat([output_fw, output_bw], axis=-1)
                    #output = tf.Print(output, [tf.shape(output)], "output=", summarize=10)

                    # shape = (batch size, max sentence length, char hidden size)
                    char_output = tf.reshape(output, shape=[-1, s[1], 2*char_hidden_size])
                    self.char_hidden_total = char_hidden_size * 2
                elif char_embedding_method == "hcnn":
                    char_outputs = []
                    # add channel
                    char_embeddings_with_channel = tf.expand_dims(char_embeddings, -1)
                    for i, kernel_dim in enumerate(kernels):
                        reduced_length = self.max_word_len - kernel_dim + 1
                        with tf.name_scope("conv-maxpool-%s" % kernel_dim):
                            # Convolution Layer
                            filter_shape = [kernel_dim, char_embedding_size, 1, char_hidden_size]
                            char_cnn_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            char_cnn_b = tf.Variable(tf.constant(0.1, shape=[char_hidden_size]), name="b")
                            conv = tf.nn.conv2d(
                                char_embeddings_with_channel,
                                char_cnn_W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
                            # Apply nonlinearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, char_cnn_b), name="relu")
                            # Maxpooling over the outputs, only on height
                            pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, reduced_length, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name="pool")
                            char_outputs.append(pooled)

                    # Combine all the pooled features
                    self.char_hidden_total = char_hidden_size * len(kernels)
                    char_hidden = tf.concat(char_outputs, 3)
                    char_output = tf.reshape(char_hidden, [-1, s[1], self.char_hidden_total])
                elif char_embedding_method == "vcnn":
                    char_embeddings_with_channel = tf.expand_dims(char_embeddings, -1)
                    char_input = char_embeddings_with_channel
                    for i, kernel_dim in enumerate([2]):
                        reduced_length = self.max_word_len - kernel_dim + 1
                        with tf.name_scope("conv-maxpool-%s" % kernel_dim):
                            # Convolution Layer
                            if i == 0:
                                filter_shape = [kernel_dim, char_embedding_size, 1, char_hidden_size]
                            else:
                                filter_shape = [kernel_dim, 1, char_hidden_size, char_hidden_size]

                            char_cnn_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            char_cnn_b = tf.Variable(tf.constant(0.1, shape=[char_hidden_size]), name="b")
                            conv = tf.nn.conv2d(
                                char_input,
                                char_cnn_W,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
                            # Apply nonlinearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, char_cnn_b), name="relu")
                            # Maxpooling over the outputs, only on height
                            pooled = tf.nn.max_pool(
                                h,
                                ksize=[1, reduced_length, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding='VALID',
                                name="pool")
                            char_input = pooled
                    char_output = char_input
                    char_output = tf.Print(char_output, [tf.shape(char_output)], "output=", summarize=10)

                    # Combine all the pooled features
                    self.char_hidden_total = char_hidden_size
                    char_output = tf.reshape(char_output, [-1, s[1], self.char_hidden_total])

                if use_char_attention: #use_char_attention:

                    # see here: http://www.marekrei.com/blog/attending-to-characters-in-neural-sequence-labeling-models/

                    # Change h* to m via another feedforward network
                    char_output = tf.reshape(char_output, [-1, self.char_hidden_total])
                    word_embeddings = tf.reshape(word_embeddings, [-1, word_embedding_size])

                    wm = tf.get_variable(
                                    initializer=tf.random_normal([self.char_hidden_total, word_embedding_size], stddev=0.1),
                                    name="charword_W", dtype=tf.float32)
                    bm = tf.get_variable(initializer=tf.zeros_initializer(), shape=[word_embedding_size],
                                         name="charword_b", dtype=tf.float32)

                    char_word = tf.matmul(char_output, wm) + bm

                    # Char Attention Here
                    with tf.variable_scope("chars_attention"):
                        # Attention mechanism
                        attention_evidence_tensor = tf.concat([word_embeddings, char_word], axis=-1)

                        w1 = tf.get_variable(initializer=tf.random_normal([word_embedding_size * 2, word_embedding_size], stddev=0.1),
                                             name="attention_W1", dtype=tf.float32)
                        b1 = tf.get_variable(initializer=tf.zeros_initializer(), shape=[word_embedding_size], name="attention_b1", dtype=tf.float32)
                        attention_output = tf.tanh(tf.matmul(attention_evidence_tensor, w1) + b1, name="attention_tanh")

                        w2 = tf.get_variable(initializer=tf.random_normal([word_embedding_size, word_embedding_size], stddev=0.1),
                                             name="attention_W2", dtype=tf.float32)
                        b2 = tf.get_variable(initializer=tf.zeros_initializer(), shape=[word_embedding_size], name="attention_b2", dtype=tf.float32)
                        attention_output = tf.sigmoid(tf.matmul(attention_output, w2) + b2, name="attention_sigmoid")
                        word_embeddings = word_embeddings * attention_output + char_word * (1.0 - attention_output)
                        word_embeddings = tf.reshape(word_embeddings,
                                                     [-1, s[1], word_embedding_size])
                else:
                    word_embeddings = tf.concat([word_embeddings, char_output], axis=-1)
                    word_embeddings = tf.reshape(word_embeddings, [-1, s[1], word_embedding_size + self.char_hidden_total])
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(lstm_hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(lstm_hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                        cell_bw,
                                                                        self.word_embeddings,
                                                                        sequence_length=self.sentences_lengths,
                                                                        dtype=tf.float32)
            lstm_output = tf.concat([output_fw, output_bw], axis=-1)

            #lstm_output = tf.Print(lstm_output, [tf.shape(lstm_output)], "lstm_output=", summarize=10)

            lstm_output = tf.nn.dropout(lstm_output, self.dropout)

        with tf.variable_scope("fc"):
            softmax_W = tf.get_variable("softmax_w",
                                shape=[2 * lstm_hidden_size, self.num_tags],
                                dtype=tf.float32)

            softmax_b = tf.get_variable("softmax_b",
                                shape=[self.num_tags],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            num_time_steps = tf.shape(lstm_output)[1]
            lstm_output = tf.reshape(lstm_output, [-1, 2 * lstm_hidden_size])
            #lstm_output = tf.Print(lstm_output, [tf.shape(lstm_output)], "lstm_output=", summarize=10)
            pred = tf.matmul(lstm_output, softmax_W) + softmax_b
            #pred = tf.Print(pred, [tf.shape(pred)], "pred=", summarize=10)
            self.logits = tf.reshape(pred, [-1, num_time_steps, self.num_tags]) # B T O (20 * 48, 24)
            #self.logits = tf.Print(logits, [tf.shape(logits)], "logits_output=", summarize=10)

        if not use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        if use_crf:
            #self.sentences_lengths = tf.Print(self.sentences_lengths, [self.sentences_lengths], "s=", summarize=10)

            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.tags, self.sentences_lengths, transition_params=self.transition_params)

            #self.transition_params = tf.Print(self.transition_params, [self.transition_params], "trans_params=", summarize=24)
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

        if load_model:
            self.sess = self.load_model()
        else:
            self.sess = None

    def add_tdnn(self, char_embeddings, embed_dim,
                 kernels=[1,2,3,4,5,6,7], feature_maps=[50, 100, 150, 200, 200, 200, 200]):
        """Time-delayed Nueral Network (cf. http://arxiv.org/abs/1508.06615v4)
        """
        # dim = [B, T, W, E] batch, seqlen, wordlen, char_emb
        shape = tf.shape(char_embeddings)
        with tf.variable_scope("tdnn"):
            layers = []
            # add channel
            char_embeddings_with_channel = tf.expand_dims(char_embeddings, -1)
            for i, kernel_dim in enumerate(kernels):
                reduced_length = self.max_word_len - kernel_dim + 1
                with tf.name_scope("conv-maxpool-%s" % i):
                    # Convolution Layer
                    filter_shape = [kernel_dim, char_embedding_size, 1, char_hidden_size]
                    char_cnn_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    char_cnn_b = tf.Variable(tf.constant(0.1, shape=[char_hidden_size]), name="b")
                    conv = tf.nn.conv2d(
                        char_embeddings_with_channel,
                        char_cnn_W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, char_cnn_b), name="relu")
                    # Maxpooling over the outputs, only on height
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, reduced_length, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    layers.append(pooled)

            # Combine all the pooled features
            cnn_hidden_total = char_hidden_size * len(kernels)
            cnn_hidden = tf.concat(layers, 3)
            cnn_hidden_flat = tf.reshape(cnn_hidden, [-1, shape[1], cnn_hidden_total])

    def load_model(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(self.model_dir, "model"))
        return sess

    def train(self, train, dev, tags, result_filename):
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        with tf.Session() as sess:
            nepoch_no_imprv = 0
            sess.run(tf.global_variables_initializer())
            if reload_model:
                print("Reloading the latest trained model...")
                saver.restore(sess, os.path.join(self.model_dir, "model"))
            for epoch in range(num_epochs):
                print("Epoch {:} out of {:}".format(epoch + 1, num_epochs))

                # num_instance = len(train)
                train_loss = 0.0
                for i, (words, labels) in enumerate(mini_batch(train, batch_size)):
                    fd, _ = self.get_feed_dict(words, labels, self.learning_rate, dropout)
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    _, train_loss = sess.run([self.train_op, self.loss], feed_dict=fd)
                #print()
                sys.stdout.write("\n")
                sys.stdout.flush()
                acc, f1 = self.evaluate(sess, dev, tags, result_filename)
                print("# loss {:04.8f} acc {:04.2f} f1 {:04.2f}".format(train_loss, 100*acc, 100*f1))

                # decay learning rate
                self.learning_rate *= learning_rate_decay
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
                viterbi_sequence, viterbi_score = viterbi_decode(logit, transition_params, self.invalid_transitions)
                #print("viterbi_seq=", viterbi_sequence, viterbi_score)
                viterbi_sequences += [viterbi_sequence]
            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=feed_dict)

            return labels_pred, sequence_lengths

    def predict(self, input_seq, word_vocab, char_vocab, label_vocab):
        words, chars = [], []
        for word in input_seq:
            char = list(word)
            word = word_vocab.encode(word)
            char = char_vocab.encode(char)
            words.append(word)
            chars.append(char)
        words = [[words, chars]]

        def pred(sess_, seq):
            labels_pred, sequence_lengths = self.predict_batch(sess_, seq)
            return [label_vocab.decode(label) for label in labels_pred[0]]

        if self.sess is None:
            self.sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.sess, os.path.join(self.model_dir, "model"))
            return pred(self.sess, words)
        else:
            return pred(self.sess, words)

    def close(self):
        if self.sess:
            self.sess.close()
            self.sess = None

    def test(self, test, tags, result_filename):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Testing model over test set")
            saver.restore(sess, os.path.join(self.model_dir, "model"))
            acc, f1 = self.evaluate(sess, test, tags, result_filename)
            print("# test acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))

    def evaluate(self, sess, test, tags, result_filename):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        prf = collections.defaultdict(lambda: {'tp' : 0, 'ans' : 0, 'act': 0})
        for words, labels in mini_batch(test, batch_size):
            labels_pred, sequence_lengths = self.predict_batch(sess, words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, tags))
                for lab_chunk in lab_chunks:
                    prf[lab_chunk[0]]['act'] += 1
                lab_pred_chunks = get_chunks(lab_pred, tags)
                lab_pred_chunks = set(lab_pred_chunks)
                for lab_pred_chunk in lab_pred_chunks:
                    prf[lab_pred_chunk[0]]['ans'] += 1
                lab_corr_chunks = lab_chunks & lab_pred_chunks
                for lab_corr_chunk in lab_corr_chunks:
                    prf[lab_corr_chunk[0]]['tp'] += 1
                correct_preds += len(lab_corr_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
        with open(result_filename, "w") as out_file:
            ttp, tans, tact = 0, 0, 0
            for label in prf:
                tp = prf[label]['tp']
                ans = prf[label]['ans']
                act = prf[label]['act']
                p = prf[label]['tp'] / float(prf[label]['ans']) if prf[label]['ans'] > 0 else 0.0
                r = prf[label]['tp'] / float(prf[label]['act']) if prf[label]['act'] > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                label = label + "    " if len(label) == 2 else label
                label = label + "  " if len(label) == 3 else label
                print("{} {} {} {} {:04.2f} {:04.2f} {:04.2f}".format(label, tp, ans, act, p*100.0, r*100.0, f1*100.0))
                print("{} {} {} {} {:04.2f} {:04.2f} {:04.2f}".format(label, tp, ans, act, p*100.0, r*100.0, f1*100.0), file=out_file)
                ttp += tp
                tans += ans
                tact += act

            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
            print(
                "{} {} {} {} {:04.2f} {:04.2f} {:04.2f}".format("Overall", ttp, tans, tact, p * 100.0, r * 100.0, f1 * 100.0))
            print(
                "{} {} {} {} {:04.2f} {:04.2f} {:04.2f}".format("Overall", ttp, tans, tact, p * 100.0, r * 100.0, f1 * 100.0),
                file=out_file)
            acc = np.mean(accs)
            return acc, f1


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        words, chars = zip(*words)
        words, sentences_lengths = pad_sentence(words, 0)
        #print(len(words[0]))
        chars, word_lengths = pad_chars(chars, pad_tok=0,
                                        max_word_length=self.max_word_len)

        feed = {
            self.words: words,
            self.sentences_lengths: sentences_lengths,
            self.chars: chars,
            self.word_lengths: word_lengths
        }

        if labels is not None:
            labels, _ = pad_sentence(labels, 0)
            #print(len(labels[0]))
            feed[self.tags] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sentences_lengths


def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag.get(tok, OUTSIDE)
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
        x_batch += [[list(x_) for x_ in x]]
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

    parser.add_option('-f', '--folds', dest='num_folds', type=int, help='run crossvalidation')
    (options, args) = parser.parse_args()

    if len(args) < 1:
        parser.error("Invalid number of argument.")

    input_dir = args[0]

    if options.build:
        build(input_dir)
    elif options.num_folds:
        num_folds = int(options.num_folds)
        run_folds(input_dir, num_folds)
    else:
        run(input_dir)


def run_folds(input_dir, num_folds):
    print("run %s folds cross validation..." % num_folds)
    for fold_num in range(num_folds):
        run_fold(input_dir, fold_num)


def run_fold(input_dir, fold_num):
    fold_dir = "fold_%s" % fold_num
    input_dir = os.path.join(input_dir, fold_dir)
    print("build...")
    build(input_dir)
    run(input_dir)



def run(input_dir):

    model_dir = os.path.join(input_dir, "model")
    
    train_filename, valid_filename, test_filename = get_input_filenames(input_dir)
    word_vocab_filename, char_vocab_filename, label_vocab_filename = get_vocab_filenames(input_dir)

    char_vocab = Vocab(char_vocab_filename, encode_char=True)
    word_vocab = Vocab(word_vocab_filename)
    tag_vocab = Vocab(label_vocab_filename, encode_tag=True)

    invalid_transitions = []
    for label1 in tag_vocab.encoding_map.keys():
        for label2 in tag_vocab.encoding_map.keys():
            if label1 == OUTSIDE:
                if label2[0] == "I":
                    invalid_transition = [tag_vocab.encode(label1), tag_vocab.encode(label2)]
                    invalid_transitions.append(invalid_transition)
            elif label2[0] == "I" and label2[2:] != label1[2:]:
                invalid_transition = [tag_vocab.encode(label1), tag_vocab.encode(label2)]
                invalid_transitions.append(invalid_transition)

    train = BIOFileLoader(train_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)
    dev = BIOFileLoader(valid_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)
    test = BIOFileLoader(test_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)

    num_words = len(word_vocab)
    num_chars = len(char_vocab)
    num_labels = len(tag_vocab)

    max_sentence_len = train.max_length()
    #max_sentence_len = 100

    max_word_len = min(train.max_word_length(), 20)

    print("max_sentence={}".format(max_sentence_len))
    print("max_word={}".format(max_word_len))

    word_embeddings_npz_filename = os.path.join(input_dir, "word.npz")
    char_embeddings_npz_filename = os.path.join(input_dir, "char.npz")
    word_embeddings = load_embeddings(word_embeddings_npz_filename)
    char_embeddings = load_embeddings(char_embeddings_npz_filename)

    #model = Model(num_words, num_labels, num_chars, max_sentence_len, max_word_len, model_dir,
    #              word_embeddings=word_embeddings, char_embeddings=char_embeddings, invalid_transitions=invalid_transitions)

    model = Model(num_words, num_labels, num_chars, max_sentence_len, max_word_len, model_dir,
                  invalid_transitions=invalid_transitions)


    vocab_tags = tag_vocab.encoding_map

    valid_result_filename = os.path.join(input_dir, "valid_res.txt")
    model.train(train, dev, vocab_tags, valid_result_filename)

    test_result_filename = os.path.join(input_dir, "test_res.txt")
    model.test(test, vocab_tags, test_result_filename)

def build(input_dir):
    embeddings_dirname = "."
    working_dir = input_dir 
    train_filename, valid_filename, test_filename = get_input_filenames(input_dir)
    word_vocab_filename, char_vocab_filename, label_vocab_filename = get_vocab_filenames(working_dir)
    word_embeddings_filename = os.path.join(embeddings_dirname, "word.txt")
    char_embeddings_filename = os.path.join(embeddings_dirname, "char.txt")
    word_embeddings_npz_filename = os.path.join(working_dir, "word.npz")
    char_embeddings_npz_filename = os.path.join(working_dir, "char.npz")


    char_vocab = Vocab(encode_char=True)
    word_vocab = Vocab()
    label_vocab = Vocab(encode_tag=True)

    train = BIOFileLoader(train_filename)
    valid = BIOFileLoader(valid_filename)
    test = BIOFileLoader(test_filename)

    char_vocab.encode_datasets([train, valid])
    word_vocab.encode_datasets([train, valid])
    label_vocab.encode_datasets([train, valid])

    vocab = read_embeddings_vocab(word_embeddings_filename)
    word_vocab.update(vocab)
    save_word_embeddings(word_embeddings_filename, word_embeddings_npz_filename, word_vocab.encoding_map)

    vocab = read_embeddings_vocab(char_embeddings_filename)
    char_vocab.update(vocab)
    save_char_embeddings(char_embeddings_filename, char_embeddings_npz_filename, char_vocab.encoding_map)

    print("word vocab size {}".format(len(word_vocab)))
    print("char vocab size {}".format(len(char_vocab)))
    print("labal vocab size {}".format(len(label_vocab)))

    char_vocab.save(char_vocab_filename)
    word_vocab.save(word_vocab_filename)
    label_vocab.save(label_vocab_filename)


if __name__ == "__main__":
    main()

