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
from BIODocument import BIODocument

SBD = SentenceBoundaryDetector()
key = "zbRJ%)l:>T}F<YW4^oVCj[.QH5yX5#`F"


class AnnotationFileLoader:
    def __init__(self, filename):
        self.docname = filename[filename.rfind("/") + 1:]
        self.document = Document(DocumentFileLoader(filename, key))
        self.sentences = self.__construct_sentences()
        self.text = self.document.text
        self.bio_document = BIODocument(self.docname, self.text, self.sentences)
        
    def __is_overlap(self, ext1, ext2):
        (start1, end1) = ext1
        (start2, end2) = ext2
        if start1 <= start2 and end1 >= end2:
            return True
        return False
    
    def __get_instances_for_sentence(self, start, end, text):
        instances = []
        for tag_type in self.document.annotation_set.types():
            #tag_type1 = tag_type.replace(" ", "_")
            if tag_type.startswith("__"): # and tag_type... Here for excluding
                continue
            for extent in self.document.annotation_set.extents_by_type(tag_type):
                if self.__is_overlap((start, end), (extent.start, extent.end)):
                    instance = Instance(extent.start, extent.end, tag_type, text[extent.start:extent.end])
                    instances.append(instance)
        instances.sort()
        return instances
    
    def __construct_sentences(self):
        sentences = []
        for i, (text, (start, end)) in enumerate(SBD.getSentences(self.document.text)):
            if not text.strip():
                continue
            instances = self.__get_instances_for_sentence(start, end, self.document.text)
            sentence = AnnotationSentence(self.docname, i, start, end, instances, self.document.text)
            sentences.append(sentence)
        return sentences
    
    def get_document(self):
        return BIODocument(self.docname, self.text, self.sentences)




def test():
    filename = sys.argv[1]
    afl = AnnotationFileLoader(filename)
    document = afl.get_document()


