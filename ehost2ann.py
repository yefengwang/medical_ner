import sys
import os
import codecs
import re

from xml.dom import minidom
from optparse import OptionParser

from Phoenix.Elements.Document import Document
from Phoenix.Elements.AnnotationSet import AnnotationSet
from Phoenix.Elements.Annotation import Annotation
from Phoenix.Elements.Extent import Extent

KNOWATOR_FILE_EXTENSION = ".knowtator.xml"
ANNOTATION_FILE_EXTENSION = ".ann"
CORPUS_DIRNAME = "corpus"
SAVED_DIRNAME = "saved"

program_name = os.path.basename(__file__)

CLASS_MENTION = 'CLASS'
ASSERTION_MENTION = 'ASSERTION'
RELATION_MENTION = 'RELATION'


def read_annotation(filename):

    xmldoc = minidom.parse(filename)

    annotations = {}
    relations = {}

    mentions = {}
    class_mentions = {}
    complex_mentions = {}
    string_mentions = {}

    mention_nodes = xmldoc.getElementsByTagName("classMention")
    for mention_node in mention_nodes:
        mention_id = mention_node.attributes['id'].value
        mention_class_node = mention_node.getElementsByTagName("mentionClass")[0]
        mention_class = mention_class_node.attributes['id'].value
        mention_text = mention_class_node.firstChild.wholeText

        mention_slots = []
        slot_mention_nodes = mention_node.getElementsByTagName("hasSlotMention")
        for slot_mention_node in slot_mention_nodes:
            slot_mention_id = slot_mention_node.attributes['id'].value
            mention_slots.append(slot_mention_id)

        mentions[mention_id] = (CLASS_MENTION, mention_class, mention_text, mention_slots)
        class_mentions[mention_id] = mentions[mention_id]

    mention_nodes = xmldoc.getElementsByTagName("stringSlotMention")
    for mention_node in mention_nodes:
        mention_id = mention_node.attributes['id'].value
        mention_slot_node = mention_node.getElementsByTagName("mentionSlot")[0]
        property_name = mention_slot_node.attributes['id'].value
        string_slot_mention_value_node = mention_node.getElementsByTagName("stringSlotMentionValue")[0]
        property_value = string_slot_mention_value_node.attributes['value'].value
        mentions[mention_id] = (ASSERTION_MENTION, property_name, property_value)
        string_mentions[mention_id] = mentions[mention_id]

    mention_nodes = xmldoc.getElementsByTagName("complexSlotMention")
    for mention_node in mention_nodes:
        mention_id = mention_node.attributes['id'].value
        mention_slot_node = mention_node.getElementsByTagName("mentionSlot")[0]
        relation_name = mention_slot_node.attributes['id'].value
        string_slot_mention_value_node = mention_node.getElementsByTagName("complexSlotMentionValue")[0]
        target_id = string_slot_mention_value_node.attributes['value'].value
        mentions[mention_id] = (RELATION_MENTION, relation_name, target_id)
        complex_mentions[mention_id] = mentions[mention_id]

    annotation_nodes = xmldoc.getElementsByTagName('annotation')
    for annotation_node in annotation_nodes:
        start = annotation_node.getElementsByTagName('span')[0].attributes['start'].value
        end = annotation_node.getElementsByTagName('span')[0].attributes['end'].value
        mention_id = annotation_node.getElementsByTagName('mention')[0].attributes['id'].value

        if mention_id in mentions:
            mention = mentions[mention_id]
            ann_type = mention[1]
            ann_text = mention[2]
            ann_type = re.sub(r'[A-Za-z]+', '', ann_type)
            if ann_type == '部位':
                ann_type = '身体部位'
            if ann_type == '治疗方法':
                ann_type = '治疗'
            annotations[mention_id] = (start, end, ann_type, ann_text, {})

    for mention_id, mention in class_mentions.items():
        mention_type, mention_class, mention_text, mention_slots = mention
        for mention_slot_id in mention_slots:
            another_mention = mentions[mention_slot_id]
            if another_mention[0] == ASSERTION_MENTION:
                assertion_name = another_mention[1]
                assertion_value = another_mention[2]
                annotations[mention_id][4][assertion_name] = assertion_value
            elif another_mention[0] == RELATION_MENTION:
                relation_id = mention_slot_id
                relation_type = another_mention[1]
                target_id = another_mention[2]
                target_annotation = annotations[target_id]
                source_annotation = annotations[mention_id]
                relations[relation_id] = [relation_type, mention_id, target_id]

    return annotations, relations


def read_project(project_dirname, verbose=True):
    if not os.path.exists(project_dirname) or not os.path.isdir(project_dirname):
        raise IOError("%s: No such directory" % project_dirname)

    corpus_dirname = os.path.join(project_dirname, CORPUS_DIRNAME)
    ann_dirname = os.path.join(project_dirname, SAVED_DIRNAME)

    if not os.path.exists(corpus_dirname) or not os.path.isdir(corpus_dirname):
        raise IOError("%s: Not an eHOST project\n" % project_dirname)

    documents = []
    for filename in os.listdir(corpus_dirname):
        corpus_filename = os.path.join(corpus_dirname, filename)
        ann_filename = os.path.join(ann_dirname, filename) + KNOWATOR_FILE_EXTENSION
        text = codecs.open(corpus_filename, encoding="utf-8").read()
        if not os.path.exists(ann_filename):
            continue
        annotations, relations = read_annotation(ann_filename)
        annotation_set = AnnotationSet()
        for mention_id, (start, end, ann_type, ann_text, assertions) in annotations.items():
            annotation = Annotation(ann_type, Extent(int(start), int(end)), id=mention_id)
            for assertion_name, assertion_value in assertions.items():
                annotation[assertion_name] = assertion_value
            annotation_set.add(annotation)

        relation_ids = []
        for mention_id, relation in relations.items():
            rel_type, annotation_id1, annotation_id2 = relation
            annotation_set.add_relation_by_id(rel_type, annotation_id1, annotation_id2, mention_id)
            relation_ids.append(mention_id)

        document = Document(text=text, annotation_set=annotation_set)
        documents.append((filename, document))
        if verbose:
            print(filename, len(annotations), len(relations))
    return documents


def convert(project_dirname, output_dirname, verbose=True):
    if not os.path.exists(project_dirname) or not os.path.isdir(project_dirname):
        sys.stderr.write("%s: %s: No such directory\n" % (program_name, project_dirname))
        sys.exit(1)

    corpus_dirname = os.path.join(project_dirname, CORPUS_DIRNAME)
    ann_dirname = os.path.join(project_dirname, SAVED_DIRNAME)

    if not os.path.exists(corpus_dirname) or not os.path.isdir(corpus_dirname):
        sys.stderr.write("%s : %s: Not an eHOST project\n" % (program_name, project_dirname))
        sys.exit(1)

    try:
        os.makedirs(output_dirname, exist_ok=True)
    except OSError as exception:
        sys.stderr.write("%s : %s: Unable to create directory\n" % (program_name, project_dirname))
        sys.exit(1)

    files = os.listdir(corpus_dirname)
    files.sort()
    for filename in files:
        corpus_filename = os.path.join(corpus_dirname, filename)
        ann_filename = os.path.join(ann_dirname, filename) + KNOWATOR_FILE_EXTENSION
        output_filename = os.path.join(output_dirname, filename) + ANNOTATION_FILE_EXTENSION
        text = codecs.open(corpus_filename, encoding="utf-8").read()
        mentions = read_annotation(ann_filename)
        annotation_set = AnnotationSet()
        for mention in mentions.values():
            annotation = Annotation(mention[2], Extent(int(mention[0]), int(mention[1])))
            annotation_set.add(annotation)
        document = Document(text=text, annotation_set=annotation_set)
        document.save_file(output_filename)


def main():
    parser = OptionParser(usage="usage: %prog source_dir target_dir",
                          version="%prog 1.0")
    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.error("wrong number of arguments")
    project_dirname, output_dirname = args[:2]
    convert(project_dirname, output_dirname)

if __name__ == "__main__":
    read_project("data/20170712")

