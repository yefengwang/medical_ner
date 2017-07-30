######################################################
class Token:
    """
    class represents a token
    """
    TEXT = 1
    STOPWORD = 2
    PUNC = 3
    SPACE = 4

    def __init__(self, textString, stemString, geniaStemString, tokenPosition, partOfSpeech, chunkTag, wordType, textPosition):
        ## text
        self.textString = textString

        ## porter stem
        self.stemString = stemString

        ## GENIA stemStemString
        self.geniaStemString = geniaStemString

        ## position of token in a segment
        self.tokenPosition = tokenPosition

        ## part of speech tag
        self.partOfSpeech = partOfSpeech

        ## chunk tag
        self.chunkTag = chunkTag

        ## type of the word
        self.wordType = wordType
        
        ## position of token in original text
        self.textPosition = textPosition
        
        self.start = self.textPosition[0]
        
        self.end = self.textPosition[1]

    def __str__(self):
        return self.textString

    def __repr__(self):
        return self.textString
    
    def __getTypeName(self, token_type):
        if token_type == self.STOPWORD:
            return "STOPWORD"
        if token_type == self.PUNC:
            return "PUNCTUATION"
        if token_type == self.TEXT:
            return "WORD"
        if token_type == self.SPACE:
            return "SPACE"
        return "UNKNOWN"
    
    def info(self):
        return "TEXT : %s :: POS : %s :: TYPE : %s" % (self.textString, self.textPosition, self.__getTypeName(self.wordType))
    
    def tup(self):
        return (self.textString, self.textPosition, self.__getTypeName(self.wordType))