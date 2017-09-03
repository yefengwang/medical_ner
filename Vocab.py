PAD = "<<PAD>>"
UNK = "<<UNK>>"
NUM = "<<NUM>>"
OUTSIDE = "O"


class Vocab:
    def __init__(self, filename=None, encode_char=False, encode_tag=False):
        self.max_idx = 0
        self.encoding_map = {}
        self.decoding_map = {}
        self.encode_char = encode_char
        self.encode_tag = encode_tag
        self._insert = True

        if filename:
            self.load(filename)
            self._insert = False
        else:
            if not self.encode_tag:
                self._encode(PAD, add=True)
                self._encode(UNK, add=True)
                if not self.encode_char:
                    self._encode(NUM, add=True)
            else:
                self._encode(OUTSIDE, add=True)

    def __len__(self):
        return len(self.encoding_map)

    def encodes(self, seq, char_level=False):
        '''
        encode a sequence
        '''
        return [self.encode(word, char_level) for word in seq]

    def encode(self, word, char_level=False):
        '''
        encode a word or a char
        '''
        if char_level:
            if self.encode_char:
                return self._encode(word, add=self._insert)
            else:
                return self._encode(word, add=self._insert)
        else:
            if self.encode_char:
                return [self._encode(char, add=self._insert) for char in word]
            else:
                return self._encode(word, add=self._insert)

    def encode_datasets(self, datasets, char_level=False):
        for dataset in datasets:
            for (xws, xcs), ys in dataset:
                if char_level:
                    if self.encode_tag:
                        self.encodes(ys)
                    elif self.encode_char:
                        self.encodes(xcs, char_level)
                    else:
                        self.encodes(xws)
                else:
                    if self.encode_tag:
                        self.encodes(ys)
                    elif self.encode_char:
                        self.encodes(xcs, char_level)
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
            idx = self.encoding_map.get(word, self.max_idx)
            if idx == self.max_idx:
                self.max_idx += 1
            self.encoding_map[word] = idx
            self.decoding_map[idx] = word
        else:
            if self.encode_tag:
                idx = self.encoding_map.get(word, self.encoding_map[OUTSIDE])
            else:
                idx = self.encoding_map.get(word, self.encoding_map[UNK])

        return idx

    def save(self, filename):
        import operator
        with open(filename, "w") as f:
            for word, idx in sorted(self.encoding_map.items(), key=operator.itemgetter(1)):
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

    def update(self, vocab):
        for word in vocab:
            self._encode(word, add=True)