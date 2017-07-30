#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:ts=4:sw=4:et:ai

"""
DocumentFileLoader class

Allows a document to be saved and loaded from a file

$Id: DocumentFileLoader.py 141 2017-07-13 08:35:38Z ywang $
"""

import codecs
import os

from Phoenix.Elements import AnnotationSet


class DocumentFileLoader(object):
    """
    Save/load a document from a file
    """

    def __init__(self, filename,  key=""):
        self.id = filename
        self.key = key
        if self.key:
            self.version = 2
        else:
            self.version = 1
            
    def load(self):
        """
        Load the text and annotations from the file.  Depending
        on the suffix of the filename, the data may be loaded in
        different ways.

        .txt - loaded as a text file with no annotations (saved as .ann)
        .ann - checked for version

        Returns (text, annotation_set)
        """
        base, ext = os.path.splitext(self.id)
        if ext == '.txt':
            return codecs.open(self.id, 'r', 'utf-8').read(), AnnotationSet.AnnotationSet()
        elif ext == '.ann':
            version,  text,  annotation_set = AnnotationSet.load(open(self.id, 'r'), self.key)
            self.version = version
            return text,  annotation_set
        else:
            raise ValueError('unknown file type')

    def save(self, text, annotation_set):
        """
        Save the annotations associated with the document
        """
        base, ext = os.path.splitext(self.id)
        if ext == '.txt':
            self.id = base + '.ann'
        with open(self.id, 'w') as f:
            AnnotationSet.dump(text, annotation_set, f, self.version, self.key)

