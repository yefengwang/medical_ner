# -*- coding: utf-8 -*-
"""Provides functions for splitting sentences and tokenising words therein
including stop-list and Porter-stemming functionality."""

import re
import os
import string

import jieba
import jieba.posseg as pseg

from Token import Token

# chars is equal to a string of alphabetical charaters and _
chars = r'[a-zA-Z_]+'

def strict_tokenise(text, char_level=False):
    """
    A better tokeniser
    @param text (str): a text string to be tokenised.
    @return (str):
    """
    words = pseg.cut(text)
    offset = 0
    for token, flag in words:

        start = offset
        tkn_len = len(token)
        offset += tkn_len
        end = offset
        if char_level:
            for p, char in enumerate(list(token)):
                yield (start + p, start + p + 1, str(p))
        else:
            yield (start, end, flag)


class Tokeniser:
    
    def __init__(self):
        """
        Tokeniser takes a stoplist and a stemmer
        """
        pass
    

    def tokenise(self, text, loose=False, offset=0, char_level=False):
        """
        Tokenise text, return list of Tokens.
        """

        tokens = []
        pos = 0
        tokenise = strict_tokenise
        for i, token_pos in enumerate(tokenise(text, char_level)):
            text_string = text[token_pos[0]:token_pos[1]]
            if text_string.strip() == "":
                continue
            stem_string = text_string
            genia_stem_string = stem_string
            pos_tag = token_pos[2]
            chunk_tag = 'O'
            
            word_type = Token.TEXT
            
            token = Token(text_string, stem_string, genia_stem_string, pos, pos_tag, chunk_tag, word_type,
                          (token_pos[0] + offset, token_pos[1] + offset))
            tokens.append(token)
            pos += 1
        return tokens
    

class TokeniserGetter(object):

    import os
    import os.path
    
    instance = None
    
    tokeniser = None
    initialised = False
    
    def __new__(cls, *args, **kargs):
        """
        make a singleton object.
        """
        if cls.instance is None:
            cls.instance = object.__new__(cls, *args, **kargs)
        return cls.instance

    def __init__(self):

        def package_home(gdict): 
            filename = gdict["__file__"] 
            return os.path.dirname(filename)
        
        def load_stoplist():
            stoplist = {}
            for row in file(os.path.join('data', 'stoplist'), "r"):
                if row and row[0]!="#":
                    stopword = row.strip()
                    stoplist[stopword] = stem(stopword)
            return stoplist

        if not self.initialised:
            #stoplist = load_stoplist()

            self.initialised = True
    
            tokeniser = Tokeniser()
            self.tokeniser = tokeniser

    @classmethod
    def getTokeniser(self):
        return TokeniserGetter().tokeniser

    #getTokeniser = classmethod(getTokeniser)

tokeniser = TokeniserGetter.getTokeniser()

def testTokeniser():
    text = u"""个人史：生于原籍,足月混合喂养，生长发育正常无长期外地居住，无血吸虫疫水接触史，正规疫苗接种史。"""
    for token in tokeniser.tokenise(text):
        print(token.info())
    
if __name__ == "__main__":
    testTokeniser()
