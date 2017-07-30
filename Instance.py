class Instance:

    def __init__(self, start, end, tag, text):
        if start < 0 or end < 0:
            raise ValueError("positions must be non-negative")
        if start > end:
            raise ValueError("start must not be after end")
        self.start = start
        self.end = end
        self.tag = tag
        self.text = text

    def __len__(self):
        return self.end - self.start
        
    def __lt__(self, other):
        return self.end <= other.start
    
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end
    
    def __gt__(self, other):
        return other < self
        
    def __str__(self):
        return str((self.start, self.end, self.tag, self.text))
        
    def __repr__(self):
        return "Instance of %s at (%s, %s): %s" % (self.tag, self.start, self.end, self.text)

    def equals(self, other, ignore_tag=False):
        fp = self == other 
        if ignore_tag:
            return fp
        else:
            return (self.tag == other.tag) and fp
        
    def left(self, other, ignore_tag=False):

        fp = self.start == other.start and self.end != other.end
        
        if ignore_tag:
            return fp
        else:
            return self.tag == other.tag and fp
        
    def right(self, other, ignore_tag=False):

        fp = (self.end == other.end) and (self.start != other.start)
        if ignore_tag:
            return fp
        else:
            return (self.tag == other.tag) and fp
    
    def meets(self, other, ignore_tag=False):

        fp = (self.start < other.start and self.end > other.start and self.end < other.end) or (self.end > other.end and self.start < other.end and self.start > other.start)
        #fp = (self.__start < other.start and self.__end > other.start and self.__end <= other.end) or (self.__end > other.end and self.__start < other.end and self.__start >=other.start)

        if ignore_tag:
            return fp
        else:
            return (self.tag == other.tag) and fp
        
    def contains(self, other, ignore_tag=False):
        """
        return True if self contains other
        """
        fp = (self.start < other.start and self.end > other.end) or (self.start < other.start and self.end > other.end)
        if ignore_tag:
            return fp
        else:
            return (self.tag == other.tag) and fp

    def coveredby(self, other, ignore_tag=False):
        """
        return True if self is covered by other
        """
        fp = (self.start > other.start and self.end < other.end) or (self.start > other.start and self.end < other.end)

        # fp = (self.__start >=other.start and self.__end < other.end) or (self.__start > other.start and self.__end <= other.end)

        if ignore_tag:
            return fp
        else:
            return (self.tag == other.tag) and fp
    
    def overlaps(self, other, ignore_tag=False):
        # fp = self.meets(other) or self.contains(other) or self.coveredby(other) or self.left(other) or self.right(other)

        fp = self.left(other) or self.right(other)

        if ignore_tag:
            return fp
        else:
            return (self.tag == other.tag) and fp

    def excludes(self, other, ignore_tag=False):
        return self.end <= other.start or self.start >= other.end



