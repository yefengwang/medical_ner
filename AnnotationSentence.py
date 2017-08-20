from Tokeniser import tokeniser

def _tokenise_segment(segment):
    (start, end), text, tag_type = segment

    bio_tag = "O"
    if tag_type:
        bio_tag = tag_type.replace(" ", "_")
    index = 0
    for t in tokeniser.tokenise(text):
        if text[t.start:t.end].strip():
            if bio_tag != "O":
                if index == 0:
                    token = [(t.start+start, t.end+start), text[t.start:t.end].strip(), t.partOfSpeech, "B-"+bio_tag.upper()]
                    yield token
                else:
                    token = [(t.start+start, t.end+start), text[t.start:t.end].strip(), t.partOfSpeech, "I-"+bio_tag.upper()]
                    yield token
            else:
                token = [(t.start+start, t.end+start), text[t.start:t.end].strip(), t.partOfSpeech, "O"]
                yield token
            index += 1


class AnnotationSentence:

    def __init__(self, doc_name, sent_num, start, end, annotations, text):
        if type(text) == str:
            self.annotations = annotations
            self.num = sent_num
            self.annotations = annotations
            self.start = start
            self.end = end
            self.tokens = []
            self.BIOTokens = []
            self._text2bio(text)
            self.doc_name = doc_name

        elif type(text) == list:
            self.annotations = annotations
            self.num = sent_num
            self.annotations = annotations
            self.start = start
            self.end = end
            self.tokens = text
            self.doc_name = doc_name

    def update(self, tokens):
        if not self._check_tokens(tokens):
            return False
        
        for (old_tkn, new_tkn) in zip(tokens, self.tokens):
            print((old_tkn, new_tkn))
            
    def _check_tokens(self, tokens):
        if len(tokens) != len(self.tokens):
            return False
        for (token1, token2) in zip(self.tokens, tokens):
            if token1[1] != token2[0]:
                return False
        return True
            
    def __str__(self):
        return self.__repr__()
        
    def __repr__(self):
        return "%s: (%s, %s) %s" % (self.num, self.start, self.end, " ".join([token[1] for token in self.tokens]))

    def _text2segments(self, text):
        index = self.start
        for instance in self.annotations:
            yield (((index, instance.start), text[index: instance.start], None))
            yield (((instance.start, instance.end), text[instance.start: instance.end], instance.tag))
            index = instance.end
        yield (((index, self.end), text[index:self.end], None))

    def _text2bio(self, text):
        for segment in self._text2segments(text):
            for token in _tokenise_segment(segment):
                self.tokens.append(token)
                self.BIOTokens.append([token[1], token[2]])


