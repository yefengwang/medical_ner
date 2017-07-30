#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The simpletokeniser moduel provides a tokenise function with other necessary
data structures for tokenising given text segments. This simple tokeniser breaks
text only by white space and newline. Notice that this simple tokeniser works
properly only with UNIX/UNIX-like format (LF only).

$Id$
"""

CHARLIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z', ',', '.', '/', '?', '<', '>', '{', '}',
            '[', ']', '(', ')', '|', '\\', '1', '2', '3', '4', '5', '6',
            '7', '8', '9', '0', '!', '~', '`', '@', '#', '$', '%', '^',
            '&', '*', '-', '=', '_', '+', ':', ';', '\'', '"']

class Token:
    def __init__(self, start, end):
        """
        @param start: integer, the starting extent
        @param end: integer, the ending extent
        """
        self.start = start
        self.end = end
        
def tokenise(text):
    """
    
    @param text: string, a free-text string to be tokenised.
    @return list: a list of Token objects
    """
    tokens = []
    length = len(text)
    start = 0
    end = 0
    prev = None
    while end < length:
        curr = text[end]
        if curr in (' ', '\n', '\t', '\x0c', '\x1b9'):
            if text[start:end] != "":
                tokens.append(Token(start, end))
                start = end
            if curr == '\n':
                tokens.append(Token(end, end + 1))
                start += 1
                #print text[end:end+1]               
        elif curr in CHARLIST and prev in ('\n', ' ', '\t', '\x0c', '\x1b9'):
            if text[start:end] != "":
                tokens.append(Token(start, end))
                start = end
        else:
            pass
        if curr == '\n':
            prev = '\n'
        prev = curr
        end += 1
    tokens.append(Token(start, end))
    return tokens

def test():
    doc = """that is very good  \n<Title>CLINICAL CHEMISTRY</Title> """
    tokens = tokenise(doc)
    for t in tokens:
    #    print t.start, t.end,
        print [doc[t.start:t.end]],
    
if __name__ == "__main__":
    test()
    