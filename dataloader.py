

UNK = "<<UNK>>"
NUM = "<<NUM>>"
NONE = "O"




def mini_batches(dataset, minibatch_size):
    x_batch, y_batch = [], []
    for (x, y) in dataset:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        x_batch.append(x)
        y_batch.append(y)

    if len(x_batch) != 0:
        yield x_batch, y_batch



def main():
    char_vocab = Vocab("char_vocab.txt", encode_char=True)
    word_vocab = Vocab("word_vocab.txt")
    tag_vocab = Vocab("tag_vocab.txt", encode_tag=True)

    train_filename = "data/conll2003/en/train.txt"
    valid_filename = "data/conll2003/en/valid.txt"
    test_filename = "data/conll2003/en/test.txt"

    train = BIOFileLoader(train_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)
    valid = BIOFileLoader(valid_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)
    test = BIOFileLoader(test_filename, word_vocab=word_vocab, char_vocab=char_vocab, tag_vocab=tag_vocab)
    for words, chars, tags in train:
        print([word_vocab.decode(word) for word in words])


    '''
    char_vocab.encode_datasets([train, valid])
    word_vocab.encode_datasets([train, valid])
    tag_vocab.encode_datasets([train, valid, test])

    char_vocab.save("char_vocab.txt")
    word_vocab.save("word_vocab.txt")
    tag_vocab.save("tag_vocab.txt")
    '''

if __name__ == "__main__":
    main()
