#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:ts=4:sw=4:et:ai

"""
AnnotationSet class

$Id: AnnotationSet.py 242 2017-07-13 06:04:00Z ywang $
"""

import sys, urllib
import json
from io import StringIO

from collections import defaultdict


from Phoenix.Elements.Annotation import Annotation
from Phoenix.Elements.Extent import Extent
from Phoenix.Elements.Relation import Relation

ANNOTATION_PREFIX = "Annotation_"


class AnnotationSet(object):
    """
    An annotation collection
    """

    def __init__(self):
        self.annotations = {}
        self._mention_by_ids = {}
        self._max_id = 0
        self.relations = defaultdict(lambda: defaultdict(dict))
        self._relation_by_ann_ids = {}

    def get_relations(self):
        relations = []
        for rel_type in self.relations:
            sources = self.relations[rel_type]
            for source in sources:
                dests = sources[source]
                for dest in dests:
                    relations.append(dests[dest])
        return relations

    def get_relation_by_ann_id(self, src_ann_id, dst_ann_id):
        return self._relation_by_ann_ids.get((src_ann_id, dst_ann_id))


    def get_relation_ids(self):
        rel_ids = []
        for relation in self.get_relations():
            rel_ids.append(relation.id)
        return rel_ids

    def _check_annotation(self, annotation):
        try:
            if self.annotations[annotation.type][annotation.extent] is annotation:
                return True
        except ValueError:
            return True
        return False

    def get_mention(self, mention_id):
        return self._mention_by_ids.get(mention_id, None)

    def add_relation_by_id(self, type, ann_id1, ann_id2, rel_id=None):
        ann1 = self.annotation_by_id(ann_id1)
        ann2 = self.annotation_by_id(ann_id2)
        if ann1 is None:
            raise ValueError("Annotation %s does not exist" % ann_id1)
        if ann2 is None:
            raise ValueError("Annotation %s does not exist" % ann_id2)
        relation = Relation(type, ann1, ann2, rel_id)
        self.add_relation(relation)

    def add_relation(self, relation):
        if not self._check_annotation(relation.source):
            raise ValueError("Relation's source conflicts with existing annotation")
        if not self._check_annotation(relation.target):
            raise ValueError("Relation's target conflicts with existing annotation")
        self.add_mention(relation)
        if not self.has_annotation(relation.source):
            self.add(relation.source)
        if not self.has_annotation(relation.target):
            self.add(relation.target)
        self._relation_by_ann_ids[(relation.source.id, relation.target.id)] = relation
        self.relations[relation.type][relation.source][relation.target] = relation

    def __iter__(self):
            for ann_type in self.annotations:
                for extent in self.annotations[ann_type]:
                    yield self.annotations[ann_type][extent]

    def types(self):
        return self.annotations.keys()
    
    def extents_by_type(self, type_):
        if type_ in self.annotations:
            return self.annotations[type_].keys()
        return []

    def by_type(self, type_):
        return self.annotations[type_].values()

    def annotation(self, type_, extent):
        return self.annotations[type_][extent]

    def annotation_by_id(self, annotation_id):
        return self._mention_by_ids.get(annotation_id)

    def annotation_by_type_start_end(self, type_, start, end):
        extent = Extent(start, end)
        try:
            self.annotations[type_][extent]
            return self.annotations[type_][extent]
        except Exception:
            return None
    
    def has_annotation(self,  annotation):
        try:
            self.annotations[annotation.type][annotation.extent]
            return True
        except Exception:
            return False

    def add_mention(self, mention):
        if mention.id is None:
            mention.id = ANNOTATION_PREFIX + str(self._max_id + 1)
            self._max_id += 1
        if mention.id in self._mention_by_ids:
            raise ValueError("%s exists in the collection" % mention.id)
        if mention.id.startswith(ANNOTATION_PREFIX):
            try:
                mention_num = int(mention.id[len(ANNOTATION_PREFIX):])
                if mention_num >= self._max_id:
                    self._max_id = mention_num
            except ValueError:
                pass
        self._mention_by_ids[mention.id] = mention

    def add(self, annotation):
        self.add_mention(annotation)
        self.annotations.setdefault(annotation.type, {})[annotation.extent] = annotation

    def rename(self, type, new_type):
        if not self.annotations.has_key(type):
            return
        if self.annotations.has_key(new_type):
            return
        self.annotations[new_type] = self.annotations[type]
        del self.annotations[type]
        for extent in self.annotations[new_type]:
            self.annotations[new_type][extent].rename(new_type)
        
    def remove(self, annotation):
        ann = self.annotations[annotation.type][annotation.extent]
        del self.annotations[annotation.type][annotation.extent]
        del self._mention_by_ids[ann.id]

    def add_type(self, type):
        self.annotations.setdefault(type, {})

    def change_extent(self,  annotation,  start,  end):
        # check for existing annotation at the new extent
        if self.annotations[annotation.type].has_key(Extent(start, end)):
            return
        # change the extent of the annotation
        annotation = self.annotations[annotation.type][annotation.extent]
        del self.annotations[annotation.type][annotation.extent]
        annotation.extent.change_extent(start,  end)
        self.annotations[annotation.type][annotation.extent] = annotation
    
    def change_type(self,  annotation,  new_type):
        # The new type with the extent exists
        if self.annotations.has_key(new_type) and self.annotations[new_type].has_key(annotation.extent):
            return
        # change the type of the annotation
        annotation = self.annotations[annotation.type][annotation.extent]
        del self.annotations[annotation.type][annotation.extent]
        annotation.rename(new_type)
        self.annotations[annotation.type][annotation.extent] = annotation

    def remove_type(self, type_):
        if self.annotations.get(type_):
            raise ValueError("annotation exists with the type: %s" % type_)
        del self.annotations[type_]

    def clear(self):
        self.annotations = {}
        self._mention_by_ids = {}


def test():
    from Phoenix.Elements.Annotation import Annotation
    from Phoenix.Elements.Extent import Extent
    ann_set = AnnotationSet()
    ann_set.add(Annotation("Name", Extent(0, 5)))
    ann_set.add(Annotation("Name", Extent(5, 10), id="Annotation_5"))
    ann_set.add(Annotation("Name", Extent(10, 11)))
    annotation1 = ann_set.annotation_by_id("Annotation_5")
    annotation2 = ann_set.annotation_by_id("Annotation_1")
    relation = Relation("Disease-Finding", annotation1, annotation2)
    ann_set.add_relation(relation)
    for annotation in ann_set:
        print(annotation)

if __name__ == "__main__":
    test()