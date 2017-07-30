import os
import random
import sys

from optparse import OptionParser

from ehost2ann import read_project


def ehost2rel(project_dir, output_filename):
    documents = read_project(project_dir)
    with open(output_filename, "w") as out_file:
        for doc_name, document in documents:
            for sent in document.sentences:
                for (rel_id, ann1_id, ann2_id, rel_type) in  document.get_relation_pairs(sent):
                    rel_info = document.get_sentence_by_ann_ids(ann1_id, ann2_id)
                    src_ann, dst_ann, src_sent, dst_sent, src_anns, dst_anns = rel_info
                    src_ann_tokens = document.get_ann_tokens(src_sent, src_ann.id)
                    dst_ann_tokens = document.get_ann_tokens(src_sent, dst_ann.id)
                    sentence = document.get_tokenised_sentence(src_sent)
                    tokens = tokens2text(sentence)
                    ann1_text = " ".join([tokens[i] for i in src_ann_tokens])
                    ann2_text = " ".join([tokens[i] for i in dst_ann_tokens])
                    #ann1_text = document.text[src_ann.extent.start:src_ann.extent.end]
                    #ann2_text = document.text[dst_ann.extent.start:dst_ann.extent.end]

                    #print("\t".join([doc_name, rel_id, ann1_text, ann2_text,
                    #      "/".join([src_ann.type, dst_ann.type, rel_type]),
                    #      " ".join(map(str, src_ann_tokens)), " ".join(map(str, dst_ann_tokens)),  " ".join(tokens2text(sentence))]))

                    print("\t".join([doc_name, rel_id, ann1_text, ann2_text,
                          "/".join([src_ann.type, dst_ann.type, rel_type]),
                          " ".join(map(str, src_ann_tokens)), " ".join(map(str, dst_ann_tokens)),
                                     " ".join(tokens2text(sentence))]), file=out_file)


        #for rel_id in document.annotation_set.get_relation_ids():
        #    rel_instance = document.get_relation_with_sentences(rel_id)
        #    src_ann, dst_ann, src_sent, dst_sent, src_anns, dst_anns = rel_instance
        #    src_ann_tokens = document.get_ann_tokens(src_sent, src_ann.id)
        #    dst_ann_tokens = document.get_ann_tokens(dst_sent, dst_ann.id)
        #    #print(doc_name, rel_id, src_sent, dst_sent, document.text[src_ann.extent.start:src_ann.extent.end])

def tokens2text(seq):
    return [token.textString for token in seq]

def main():
    parser = OptionParser(usage="usage: %prog [option] eHOST_project_dir output_file\n"
                                "\n"
                                "extract relation from eHOST annotation project"
                                "",
                          version="%prog 1.0")

    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.print_help()
        sys.exit(1)

    project_name, output_filename = args[:2]
    ehost2rel(project_name, output_filename)



if __name__ == "__main__":
    main()
