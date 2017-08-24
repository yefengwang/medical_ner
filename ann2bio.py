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

SBD = SentenceBoundaryDetector()


def _is_overlap(ext1, ext2):
    (start1, end1) = ext1
    (start2, end2) = ext2
    if start1 <= start2 and end1 >= end2:
        return True
    return False


def _get_instances_for_sentence(document, start, end, text):
    instances = []
    for tag_type in document.annotation_set.types():
        # skip internal representation
        if tag_type.startswith("__"):
            continue
        for extent in document.annotation_set.extents_by_type(tag_type):
            if _is_overlap((start, end), (extent.start, extent.end)):
                instance = Instance(extent.start, extent.end, tag_type, text[extent.start:extent.end])
                instances.append(instance)
    instances.sort()
    return instances


def _construct_sentences(doc_name, document):
    sentences = []
    for i, (text, (start, end)) in enumerate(SBD.getSentences(document.text)):
        if not text.strip():
            continue
        instances = _get_instances_for_sentence(document, start, end, document.text)
        sentence = AnnotationSentence(doc_name, i, start, end, instances, document.text)
        sentences.append(sentence)
    return sentences


def convert_to_bio_sequence(document):
    """converts an annotation document to bio sequences
    :param document input document: 
    :return a list of sequences: 
    """
    doc_name = "" # I don't care about the document, will be added in later
    sentences = _construct_sentences(doc_name, document)
    bio_document = BIODocument(doc_name, document.text, sentences)
    return bio_document.sentences


def print_bio_sequence(sequences, delimiter="\t", file=sys.stdout):
    """print out the IOB sequences
    :param sequences: the input sequences (for documents)
    :param delimiter: separator to join the columns
    :param file: the output stream, default to stdout
    :return: None
    """

    for sentence in sequences:
        for i, [position, token, part_of_speech, tag] in enumerate(sentence.tokens):
            print("%s" % (delimiter.join([token, str(position), part_of_speech, tag])), file=file)

        if sentence.tokens:
            print(file=file)
