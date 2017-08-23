#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import sys, os
import time
import re


class SentenceBoundaryDetector:
    def __init__(self, verbose = True):
        pass
    
    def getSentences(self, text):
        """ Given a text segments, return the sentences."""
        sentences = []
        offset = 0
        for line in text.split("\n"):
            for sent in re.findall(u'[^!?。\!\?]+[!?。\!\?]?', line, flags=re.U):
                start = offset
                offset += len(sent)
                sentences.append((text[start:offset], (start, offset)))
            offset += len("\n")
        return sentences

if __name__=="__main__":
    paragraph = """热带风暴尚塔尔是2001年大西洋飓风季的一场在8月穿越了加勒比海的北大西洋热带气旋,尚塔尔于8月14日由热带大西洋的一股东风波发展而成。
其存在的大部分时间里都在快速向西移动，退化成东风波后穿越了向风群岛。"""
    #paragraph = """热带风暴尚塔尔是2001年大西洋飓风季的一场在8月穿越了加勒比海的北大西洋热带气旋。
    #尚塔尔于8月14日由热带大西洋的一股东风波发展而成，其存在的大部分时间里都在快速向西移动，退化成东风波后穿越了向风群岛。"""

    print(paragraph)
    sbd = SentenceBoundaryDetector()
    sents = sbd.getSentences(paragraph)
    for sent in sents:
        print(sent)
        pass
