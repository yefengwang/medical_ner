#! /usr/bin/env python2.5
# vim:ts=4:sw=4:et:ai

"""
Extent class

$Id: Extent.py 152 2017-07-13 14:24:30Z ywang $
"""


class Extent(object):
    """
    A class to manage a pair of textual positions: [start, end)

    >>> e = Extent(0, 5)
    >>> e.start == 0
    True
    >>> e.end == 5
    True
    >>> Extent(-5, 0)
    Traceback (most recent call last):
    ...
    ValueError: positions must be non-negative
    >>> Extent(10, 5)
    Traceback (most recent call last):
    ...
    ValueError: start must be not be after end
    >>> e0t5 = Extent(0, 5)
    >>> e0t10 = Extent(0, 10)
    >>> e5t10 = Extent(5, 10)
    >>> e == e0t5
    True
    >>> e0t5 < e5t10
    True
    >>> e5t10 > e0t5
    True
    >>> e0t10 < e0t5
    False
    """

    def __init__(self, start, end, text=None):
        if start < 0 or end < 0:
            raise ValueError('positions must be non-negative')
        if start > end:
            raise ValueError('start must be not be after end')
        self.start = start
        self.end = end
        self.text = text

    def __len__(self):
        return self.end - self.start

#   def compare(self, other):
#       """
#       Return a tuple with the results of cmp on the start and end positions
#       """
#       s, e  = self.__start, self.__end
#       os, oe = other.start, other.end
#       return tuple(map(cmp, (s, s, e, e), (os, oe, os, oe)))

    def __lt__(self, other):
        """
        Return true if self.end <= other.start
        """
        return self.end <= other.start

    def __eq__(self, other):
        """
        Return true if start and end are equal
        """
        return self.start == other.start and self.end == other.end

    def __gt__(self, other):
        """
        Return true if self.start >= other.end
        """
        return other < self

    def __hash__(self):
        return hash((self.start, self.end))

    def __str__(self):
        return "(%d, %d)" % (self.start, self.end)

    def __repr__(self):
        return "Extent(%d, %d)" % (self.start, self.end)
    
    def change_extent(self,  start,  end):
        self.start = start
        self.end = end

    def is_overlap(self, other):
        return not (other.start >= self.end or other.end < self.start)


if __name__ == "__main__":
    extent = Extent(0, 1)
    print([extent])

