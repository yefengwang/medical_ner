import os
import sys
import tensorflow as tf
import shutil
from xml.dom import minidom
import datetime

from preprocessing import prepare_document, prepare_sentence

from WordNERModel import Vocab, Model
from WordNERModel import get_vocab_filenames

from Instance import Instance


class NERClassifier:

    def __init__(self, input_dir):
        model_dir = os.path.join(input_dir, "model")

        word_vocab_filename, char_vocab_filename, label_vocab_filename = get_vocab_filenames(input_dir)

        self.char_vocab = Vocab(char_vocab_filename, encode_char=True)
        self.word_vocab = Vocab(word_vocab_filename)
        self.label_vocab = Vocab(label_vocab_filename, encode_tag=True)

        num_words = len(self.word_vocab)
        num_chars = len(self.char_vocab)
        num_labels = len(self.label_vocab)

        # max_sentence_len = train.max_length()
        max_sentence_len = 100

        max_word_len = 20

        model = Model(num_words, num_labels, num_chars, max_sentence_len, max_word_len, model_dir, load_model=True)
        self.model = model

    def predict(self, input_seq):
        return self.model.predict(input_seq, self.word_vocab, self.char_vocab, self.label_vocab)

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
                                    "".join([token[0] for token in tokens[extent_start:i]]))
                instances.append(instance)
                current_tag = None
        previous_end = end
    if current_tag:
        instance = Instance(ext_char_start, end, current_tag,
                            "".join([token[0] for token in tokens[extent_start:i + 1]]))
        instances.append(instance)
    instances.sort()
    return instances


def read_prediction(sentences):
    annotations = []
    for tokens in sentences:
        for ann in get_annotations_for_sentence(tokens):
            annotations.append((ann.start, ann.end, ann.tag, ann.text))
    return annotations

classifier = NERClassifier(sys.argv[1])

def predict_to_ehost(input_dirname, output_dirname):
    os.makedirs(output_dirname, exist_ok=True)
    os.makedirs(os.path.join(output_dirname, "corpus"), exist_ok=True)
    os.makedirs(os.path.join(output_dirname, "saved"), exist_ok=True)
    for filename in os.listdir(input_dirname):
        if not filename.endswith(".txt"):
            continue
        full_filename = os.path.join(input_dirname, filename)
        shutil.copy(full_filename, os.path.join(output_dirname, "corpus", filename))
        document = open(full_filename, "r").read()
        annotations = predict(document)
        output_xml_filename = os.path.join(output_dirname, "saved", filename + ".knowtator.xml")
        xml_doc = make_ehost_xml(document, annotations, filename, "model")
        xml_doc.writexml(open(output_xml_filename, "w"))

def make_ehost_xml(document, annotations, filename, model_name):
    doc = minidom.Document()
    annotations_node = doc.createElement("annotations")
    annotations_node.setAttribute("textSource", filename)
    for ann_id, annotation in enumerate(annotations):
        (start, end, tag, text) = annotation
        annotation_node = doc.createElement("annotation")
        mention_node = doc.createElement("mention")
        annotation_id = "Predict_%s_%s" % (filename, ann_id + 1)
        mention_node.setAttribute("id", annotation_id)
        annotator_node = doc.createElement("annotator")
        annotator_node.setAttribute("id", "model")
        annotator_node.appendChild(doc.createTextNode(model_name))
        span_node = doc.createElement("span")
        span_node.setAttribute("start", str(start))
        span_node.setAttribute("end", str(end))
        spanned_text_node = doc.createElement("spannedText")
        spanned_text = document[int(start):int(end)]
        spanned_text_node.appendChild(doc.createTextNode(spanned_text))
        creation_date_node = doc.createElement("creationDate")
        time_now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        creation_date_node.appendChild(doc.createTextNode(time_now))
        annotation_node.appendChild(mention_node)
        annotation_node.appendChild(annotator_node)
        annotation_node.appendChild(span_node)
        annotation_node.appendChild(spanned_text_node)
        annotation_node.appendChild(creation_date_node)
        annotations_node.appendChild(annotation_node)
        class_mention_node = doc.createElement("classMention")
        class_mention_node.setAttribute("id", annotation_id)
        mention_class_node = doc.createElement("mentionClass")
        mention_class_node.setAttribute("id", tag)
        mention_class_node.appendChild(doc.createTextNode(spanned_text))
        class_mention_node.appendChild(mention_class_node)
        annotations_node.appendChild(class_mention_node)
    return annotations_node



def predict(document):

    sentences = prepare_document(document)
    sents = []
    for sentence in sentences:
        tkns = []
        tokens, positions, pos, tags = zip(*sentence)
        tags = classifier.predict(tokens)
        for (token, position, tag) in zip(tokens, positions, tags):
            tkns.append((token, position, tag))
        sents.append(tkns)
    return read_prediction(sents)

if __name__ == "__main__":
    predict_to_ehost(sys.argv[2], sys.argv[3])