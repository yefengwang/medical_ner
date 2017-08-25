import numpy as np


def read_embeddings_vocab(filename):
    vocab = set()
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split('\t')
            word = line[0]
            vocab.add(word)
    return vocab


def save_word_embeddings(embedding_filename, npz_filename, vocab):
    embeddings = None
    with open(embedding_filename) as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            if i == 0:
                _, dim = line[0], int(line[1])
                embeddings = np.zeros([len(vocab), dim])
                continue

            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(npz_filename, embeddings=embeddings)

def load_embeddings(npz_filename):
    print("load embeddings {} ...".format(npz_filename))
    try:
        with np.load(npz_filename) as data:
            return data["embeddings"]

    except IOError:
        raise "Unable to load file:" + npz_filename


def save_char_embeddings(embedding_filename, npz_filename, vocab):
    embeddings = None
    with open(embedding_filename) as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            if i == 0:
                _, dim = line[0], int(line[1])
                embeddings = np.zeros([len(vocab), dim])
                continue
            word = line[0]
            embedding = [float(x) for x in line[2:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(npz_filename, embeddings=embeddings)
