import os
import sys

from AnnotationSentence import AnnotationSentence
from Instance import Instance
from BIODocument import BIODocument
from DocumentManipulater import DocumentManipulater


class PositionFileLoader:

    def __init__(self, position_file_content):
        self.documents = self.__load_position_file(position_file_content)

    def update_BIO_tags(self, output_file_content):
        BIO_tags = self.__load_BIO_file(output_file_content)
        docs = self.__update_documents(BIO_tags)
        return docs


    def __load_BIO_file(self, output_file_content):
        """
        Load the BIO files, return sentences as a list of (word, tag)
        """
        sentences = []
        sentence = []
        for line in output_file_content.split("\n"):
            if not line.strip():
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            row = line.split()
            word = row[0]
            tag = row[-1]
            sentence.append((word, tag))
        return sentences
        
    def __update_sentence_tags(self, sentence, tags):
        """
        return a new sentence with replaced tags
        """
        new_tokens = []
        for token, tag in zip(sentence.tokens, tags):
            new_token = (token[0], token[1], tag)
            new_tokens.append(new_token)
        annotations = self.__get_annotations_for_sentence(new_tokens)
        new_sentence = AnnotationSentence(sentence.docname, sentence.num, sentence.start, sentence.end, annotations, new_tokens)
        return new_sentence
    
    def __update_documents(self, tags_list):
        """
        return a list of documents, with updated sentences
        """
        docs = [] # updated documents
        # sentence_number in the tags_list
        sentence_num = 0
        for document in self.documents:
            updated_sentences = []
            for doc_sentence in document.sentences:

                # get the word, tag list for the current sentence
                # and we have to align the sentence_num for each document
                word_tag_list = tags_list[sentence_num]

                # update current sentence with the tag list
                words, tags = zip(*word_tag_list)
                updated_sentence = self.__update_sentence_tags(doc_sentence, tags)
                updated_sentences.append(updated_sentence) # add the new sentence to the sentence list

                sentence_num+=1

            # create a new document with updated sentences
            new_document = BIODocument(document.docname, document.text, updated_sentences)
            docs.append(new_document)
        return docs

    def __get_annotations_for_sentence(self, tokens):
        """
        Return entity extents, in (start, end, tagtype) token positions
        """
        instances = []
        current_tag = None
        extent_start = 0
        ext_char_start = 0
        previous_end = 0
        for i, ((start, end), token, tag) in enumerate(tokens):
            if tag[0] == "B":
                if current_tag:
                    instance = Instance(ext_char_start, previous_end, current_tag, " ".join([token[2] for token in tokens[extent_start:i]]))
                    instances.append(instance)
                current_tag = tag[2:]
                extent_start = i
                ext_char_start = start
            elif tag[0] == "I":
                pass
            elif tag[0] == "O":
                if current_tag:
                    #print tokens
                    instance = Instance(ext_char_start, previous_end, current_tag, " ".join([token[1] for token in tokens[extent_start:i]]))
                    instances.append(instance)
                    current_tag = None
            previous_end = end
        if current_tag:
            instance = Instance(ext_char_start, end, current_tag, " ".join([token[1] for token in tokens[extent_start:i+1]]))
            instances.append(instance)
        instances.sort()
        return instances
    
    def __load_position_file(self, position_file_content):
        """
        load the position file and return documents as BIODocument
        """
        documents = []
        for line in position_file_content.split("\n"):
            if not line.strip(): 
                continue
            raw_doc = eval(line.strip())
            docname, text, sents = raw_doc
            sentences = []
            for sent in sents:
                (sent_num, (sent_start, sent_end), sent_tokens) = sent
                annotations = self.__get_annotations_for_sentence(sent_tokens)
                sentence = AnnotationSentence(docname, sent_num, sent_start, sent_end, annotations, sent_tokens)
                sentences.append(sentence)
            document = BIODocument(docname, text, sentences)
            documents.append(document)
        return documents
        
def test():
    position_filename = "tmp/position"
    output_filename = "tmp/output"
    pfl = PositionFileLoader(position_filename)
    #for doc in pfl.documents:
    #    print doc.text
    docs = pfl.update_BIO_tags(output_filename)
    for document in docs:
        dm = DocumentManipulater(document)
        dm.print_text_segment()

if __name__ == "__main__":
    test()
