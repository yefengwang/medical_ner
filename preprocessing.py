"""
Converts annotation to bio format

@author Yefeng Wang
@date 2017-07-15
@version 1.0
"""
import sys

from SentenceBoundaryDetector.sbd import SentenceBoundaryDetector
from AnnotationSentence import AnnotationSentence
from Instance import Instance
from BIODocument import BIODocument
from Tokeniser import tokeniser

SBD = SentenceBoundaryDetector()


def prepare_document(document):
    sentences = []
    for i, (text, (start, end)) in enumerate(SBD.getSentences(document)):
        if not text.strip():
            continue
        sentence = []
        for t in tokeniser.tokenise(text):
            if text[t.start:t.end].strip():
                token = [text[t.start:t.end].strip(), (t.start + start, t.end + start), t.partOfSpeech, "O"]
                sentence.append(token)
        sentences.append(sentence)
    return sentences


def prepare_sentence(text):
    start = 0
    sentence = []
    for t in tokeniser.tokenise(text):
        if text[t.start:t.end].strip():
            token = [(t.start + start, t.end + start), text[t.start:t.end].strip(), t.partOfSpeech, "O"]
            sentence.append(token)
    return sentence
