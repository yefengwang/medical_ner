from Phoenix.Elements import Annotation
from Instance import Instance

def read(content):
    documents = []
    document = []
    tokens = []
    prev = (-1, 0)
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            if tokens:
                document.append(tokens)
                tokens = []
        else:
            ls = line.split(' ')
            word, pos, tag = ls[0], ls[1], ls[-1]
            start, end = pos.split(",")
            start, end = int(start), int(end)
            if start < prev[-1]:
                documents.append(document)
                document = []
            tokens.append((word, (start, end), tag))
            prev = (start, end)
    if document:
        documents.append(document)

    return documents


def get_annotations_for_sentence(tokens):
    """
    Return entity extents, in (start, end, tagtype) token positions
    """
    instances = []
    current_tag = None
    extent_start = 0
    ext_char_start = 0
    previous_end = 0
    for i, (token, (start, end), tag) in enumerate(tokens):
        if tag[0] == "B":
            if current_tag:
                instance = Instance(ext_char_start, previous_end, current_tag,
                                    " ".join([token[2] for token in tokens[extent_start:i]]))
                instances.append(instance)
            current_tag = tag[2:]
            extent_start = i
            ext_char_start = start
        elif tag[0] == "I":
            pass
        elif tag[0] == "O":
            if current_tag:
                # print tokens
                instance = Instance(ext_char_start, previous_end, current_tag,
                                    " ".join([token[0] for token in tokens[extent_start:i]]))
                instances.append(instance)
                current_tag = None
        previous_end = end
    if current_tag:
        instance = Instance(ext_char_start, end, current_tag,
                            " ".join([token[0] for token in tokens[extent_start:i + 1]]))
        instances.append(instance)
    instances.sort()
    return instances


def read_prediction(content):
    annotations = []
    for document in read(content):
        for tokens in document:
            annotations += get_annotations_for_sentence(tokens)
    return annotations
