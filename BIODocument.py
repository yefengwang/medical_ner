import sys
import os
import os.path
import random
from Phoenix.Elements.Extent import Extent
from Phoenix.Elements.Annotation import Annotation
from Phoenix.Elements.AnnotationSet import AnnotationSet
from Phoenix.Elements.Document import Document
from Phoenix.Elements.DocumentFileLoader import DocumentFileLoader
from SentenceBoundaryDetector.sbd import SentenceBoundaryDetector
from Tokeniser import tokeniser
from AnnotationSentence import AnnotationSentence
from Instance import Instance

class BIODocument:
    def __init__(self, docname, text, sentences=None):
        if sentences:
            self.docname = docname
            self.text = text
            self.sentences = sentences
        else:
            self.docname = docname
            self.text = text
            self.sentences = self.__create_sentences(text)
        
    def __create_sentences(self, doc_text):
        SBD = SentenceBoundaryDetector()
        sentences = []
        for i, (text, (start, end)) in enumerate(SBD.getSentences(doc_text)):
            if not text.strip():
                continue
            sentence = AnnotationSentence(self.docname, i, start, end, [], doc_text)
            sentences.append(sentence)
        return sentences
        
    def __str__(self):
        return self.docname
        
    def __repr__(self):
        return self.docname

