#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:ts=4:sw=4:et:ai

"""
Phoenix class

$Id: Annotation.py 166 2017-07-13 05:44:13Z ywang $
"""


class Annotation:
    """
    An annotation
    """

    def __init__(self, type, extent, properties={}, id=None):
        if not isinstance(type, str):
            raise TypeError('type must be a string type')

        self.type = type
        self.extent = extent
        self.properties = {}
        self.properties.update(properties)
        self.id = id

    def __getitem__(self, key):
        try:
            return dict.properties[key]
        except KeyError:
            raise KeyError(key)

    def get(self,  key,  default=None):
        return self.properties.get(key,  default)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError(key)
        if not isinstance(value, str):
            raise TypeError(value)
        if key not in self.properties or self.properties[key] != value:
            self.properties[key] = value

    def __delitem__(self, key):
        if not isinstance(key, str):
            raise TypeError
        try:
            del self.properties[key]
        except KeyError:
            raise KeyError(key)
    
    def rename(self, new_name):
        self.type = new_name

    def keys(self):
        return self.properties.keys()

    def __str__(self):
        return "Annotation(%s, %s, %s, %s)" % (self.type, self.extent, dict.__str__(self.properties), self.id)

    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    from Phoenix.Elements.Extent import Extent
    annotation = Annotation("疾病", Extent(0, 20))
    annotation["时间"] = "当前"
    print([annotation])
