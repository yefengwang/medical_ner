class Relation(object):

    def __init__(self, type, source, target, id=None):
        self.type = type
        self.source = source
        self.target = target
        self.id = id

    def __repr__(self):
        return '%s %s %s %s' % (self.id, self.source.type, self.target.type, self.type)
