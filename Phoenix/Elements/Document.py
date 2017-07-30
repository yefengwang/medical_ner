#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:ts=4:sw=4:et:ai

"""
Document class: Text + AnnotationSet

$Id: Document.py 141 2017-07-13 08:35:38Z ywang $
"""
import collections

from Phoenix.Elements.AnnotationSet import AnnotationSet
from Phoenix.Elements.Relation import Relation

from SentenceBoundaryDetector.sbd import SentenceBoundaryDetector
from Tokeniser import tokeniser
from Token import Token

SBD = SentenceBoundaryDetector()

NONE_TYPE = '无'

class Document(object):
    """
    The document object. May be instantiated with either some kind
    of loader/dumper or an initial Text object (and optionally an
    AnnotationSet).
    """

    def __init__(self, text, annotation_set=None):
        self.text = text
        if annotation_set is not None:
            self.annotation_set = annotation_set
        else:
            self.annotation_set = AnnotationSet()

        self.sentences = []
        self.__construct_sentences()
        self._sentences_by_annotation = {}
        self._annotations_by_sentence = collections.defaultdict(list)

        self._tokenised_sentences = {}
        self._ann_tokens = collections.defaultdict(lambda: collections.defaultdict(list))

        self.build_index()
        self.__tokenise_sentences()

    def __tokenise_sentences(self):
        for (start, end) in self.sentences:
            sent_text = self.text[start:end]
            tokens = tokeniser.tokenise(sent_text, offset=start)
            anns = self._annotations_by_sentence[(start, end)]
            i = 0
            j = 0
            if anns:
                new_tokens = []
                while j < len(tokens):
                    token = tokens[j]
                    #print("TOKEN:", token.textPosition)
                    #token.textPosition = (token.textPosition[0] + start, token.textPosition[1] + start)
                    #print("TOKEN_SHIFT:", token.textPosition)
                    if i >= len(anns):
                        new_tokens.append(token)
                        j += 1
                        continue

                    ann = self.annotation_set.annotation_by_id(anns[i])
                    #print("ANN:", i, ann.extent)
                    if token.textPosition[0] >= ann.extent.end:
                        i += 1
                        continue

                    if token.textPosition[1] <= ann.extent.start:
                        new_tokens.append(token)
                        j += 1
                        continue
                    if token.textPosition[0] >= ann.extent.end:
                        new_tokens.append(token)
                        j += 1
                        continue

                    # Tokens within annotations
                    token1, token2 = None, None
                    if token.textPosition[0] < ann.extent.start and token.textPosition[1] > ann.extent.start:
                        token1_start, token1_end = (token.textPosition[0], ann.extent.start)
                        token_text = self.text[token1_start: token1_end]
                        token1 = Token(token_text, token_text, token_text, token.tokenPosition,
                                       token.partOfSpeech, token.chunkTag, token.wordType, (token1_start, token1_end))

                        token_text = self.text[ann.extent.start: token.textPosition[1]]
                        token = Token(token_text, token_text, token_text, token.tokenPosition,
                                      token.partOfSpeech, token.chunkTag,
                                      token.wordType, (ann.extent.start, token.textPosition[1]))

                    if token.textPosition[1] > ann.extent.end and token.textPosition[0] < ann.extent.end:
                        token_text = self.text[token.textPosition[0]: ann.extent.end]
                        token = Token(token_text, token_text, token_text, token.tokenPosition,
                                      token.partOfSpeech, token.chunkTag,
                                      token.wordType, (token.textPosition[0], ann.extent.end))

                        token_text = self.text[ann.extent.end: token.textPosition[1]]
                        token2 = Token(token_text, token_text, token_text, token.tokenPosition,
                                      token.partOfSpeech, token.chunkTag,
                                      token.wordType, (ann.extent.end, token.textPosition[1]))
                    if token1:
                        new_tokens.append(token1)

                    new_tokens.append(token)
                    self._ann_tokens[(start, end)][ann.id].append(j)
                    #print(self.text[ann.extent.start:ann.extent.end])
                    #print(token)
                    #print(ann.id, i, self._ann_tokens[(start, end)])

                    if token2:
                        new_tokens.append(token2)

                    j += 1

                self._tokenised_sentences[(start, end)] = new_tokens

    def __construct_sentences(self):
        self.sentences = []
        self.sentences_map = {}
        for i, (text, (start, end)) in enumerate(SBD.getSentences(self.text)):
            if not text.strip():
                continue
            self.sentences.append((start, end))
            self.sentences_map[(start, end)] = len(self.sentences)

    def build_index(self):
        annotations = [annotation for annotation in self.annotation_set]
        annotations.sort(key=lambda annotation: annotation.extent.start)
        i = 0
        j = 0
        while j < len(annotations) and i < len(self.sentences):
            sent = self.sentences[i]
            ann = annotations[j]
            if sent[1] <= ann.extent.start:
                i += 1
                continue
            elif sent[0] <= ann.extent.start and sent[1] >= ann.extent.end:
                j += 1
                self._sentences_by_annotation[ann.id] = sent
                self._annotations_by_sentence[sent].append(ann.id)
            else:
                j += 1

    def get_assertion_with_context(self, annotation_id):
        sent = self._sentences_by_annotation[annotation_id]

    def get_sentence_by_ann_ids(self, ann1_id, ann2_id):
        src_ann = self.annotation_set.annotation_by_id(ann1_id)
        dst_ann = self.annotation_set.annotation_by_id(ann2_id)

        src_sent = self._sentences_by_annotation[ann1_id]
        dst_sent = self._sentences_by_annotation[ann2_id]

        src_anns = self._annotations_by_sentence[src_sent]
        dst_anns = self._annotations_by_sentence[dst_sent]

        return src_ann, dst_ann, src_sent, dst_sent, src_anns, dst_anns

    def get_relation_with_sentences(self, relation_id):
        relation = self.annotation_set.get_mention(relation_id)
        if relation is None or type(relation) != Relation:
            raise ValueError("Unable find relation for %s" % relation_id)
        src_ann_id = relation.source.id
        dst_ann_id = relation.target.id
        src_ann = self.annotation_set.annotation_by_id(relation.source.id)
        dst_ann = self.annotation_set.annotation_by_id(relation.target.id)

        src_sent = self._sentences_by_annotation[src_ann_id]
        dst_sent = self._sentences_by_annotation[dst_ann_id]

        src_anns = self._annotations_by_sentence[src_sent]
        dst_anns = self._annotations_by_sentence[dst_sent]
        #print(src_ann.type, dst_ann.type, relation.type)
        #print(src_sent, src_anns)

        return src_ann, dst_ann, src_sent, dst_sent, src_anns, dst_anns

    def get_tokenised_sentence(self, extent):
        return self._tokenised_sentences.get(extent)

    def get_ann_tokens(self, sent, ann_id):
        return self._ann_tokens[sent][ann_id]

    def get_anns_in_sentences(self, sent):
        anns = self._annotations_by_sentence[sent]

    def get_relation_pairs(self, sent):
        rel_pairs = []
        anns = self._annotations_by_sentence[sent]
        if len(anns) <= 1:
            return []
        for i in range(len(anns) - 1):
            for j in range(i + 1, len(anns)):
                ann1_id = anns[i]
                ann2_id = anns[j]
                #print(ann1, ann2)
                src_ann_tokens = self.get_ann_tokens(sent, ann1_id)
                dst_ann_tokens = self.get_ann_tokens(sent, ann2_id)
                ann1 = self.annotation_set.get_mention(ann1_id)
                ann2 = self.annotation_set.get_mention(ann2_id)
                #print(ann1.type, ann2.type)

                rel = self.annotation_set.get_relation_by_ann_id(ann1_id, ann2_id)
                if not rel:
                    rel = self.annotation_set.get_relation_by_ann_id(ann2_id, ann1_id)
                #print(rel)
                rel_type = None
                if rel:
                    rel_type = rel.type
                    rel_id = rel.id
                else:
                    rel_id = ann1.id + "-" + ann2.id
                    rel_type = NONE_TYPE
                rel_pairs.append((rel_id, ann1_id, ann2_id, rel_type))
        return rel_pairs







