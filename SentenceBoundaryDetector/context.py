#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

def normalizeText(text):
    newtext = []
    for l in text:
        if True == l.islower():
            newtext.append('x')
        elif True == l.isupper():
            newtext.append('X')
        elif True == l.isdigit():
            newtext.append('0')
        else:
            newtext.append(l)

    return "".join(newtext)

def isTitlePrefix(text):
    title_prefix = ('Mr.', 'Mrs.', 'Prof.', 'Dr.')

    for t in title_prefix:
        if text.startswith(t):
            return True

    else:
        return False
######################################################
def getContext00(text):
    """ the prefix and suffix of the current token"""
    ntext = normalizeText(text)
    context = ["CurText="+text, "CurNTText="+ntext]
    
    max_affixlen = min(4, len(text))
    for i in range(0, max_affixlen):
        #original text
        context.append("Prefix%i=%s" % (i, text[0:i+1]))
        context.append("Suffix%i=%s" % (i, text[len(text)-i-1:len(text)]))

        #normalized text
        context.append("PrefixNT%i=%s" % (i, ntext[0:i+1]))
        context.append("SuffixNT%i=%s" % (i, ntext[len(ntext)-i-1:len(ntext)]))

    return context


def getContext01(prev_text, prev_tag):
    """ the previous token"""
    context = ["Prev=" + prev_text, "PrevTag=" + prev_tag]

    max_affixlen = min(4, len(text))
    for i in range(0, max_affixlen):
        #original text
        context.append("PrevPrefix%i=%s" % (i, text[0:i+1]))
        context.append("PrevSuffix%i=%s" % (i, text[len(text)-i-1:len(text)]))

        #normalized text
        context.append("PrevNTPrefix%i=%s" % (i, ntext[0:i+1]))
        context.append("PrevNTSuffix%i=%s" % (i, ntext[len(ntext)-i-1:len(ntext)]))

    return context


def getContext02(next_text):
    """ the next token """
    text = next_text
    ntext = normalizeText(next_text)
    context = ["Next="+text, "NextNT="+ntext]

    max_affixlen = min(4, len(text))
    for i in range(0, max_affixlen):
        #original text
        context.append("NextPrefix%i=%s" % (i, text[0:i+1]))
        context.append("NextSuffix%i=%s" % (i, text[len(text)-i-1:len(text)]))

        #normalized text
        context.append("NextNTPrefix%i=%s" % (i, ntext[0:i+1]))
        context.append("NextNTSuffix%i=%s" % (i, ntext[len(ntext)-i-1:len(ntext)]))        
    
    return context
   

def getContext(cur_text, prev_text = None, prev_tag = None, next_text = None):
    context = []

    #features of current token
    context += getContext00(cur_text)

    #features of previous token
    try:
        context += getContext01(prev_text, prev_tag)
        #pass
    except:
        pass

    #features of next token
    try:
        context += getContext02(next_text)
        #pass
    except:
        pass

    return context
