from AnnotationSentence import AnnotationSentence
from Instance import Instance
from BIODocument import BIODocument

class DocumentManipulater:
    
    def __init__(self, document):
        self.__document = document
    
    def __get_text_segment(self, sentence):
        index = sentence.start
        for instance in sentence.annotations:
            if self.__document.text[index: instance.start].strip():
                yield ((index, instance.start), self.__document.text[index: instance.start], "O")
            yield ((instance.start, instance.end), self.__document.text[instance.start: instance.end], instance.tag)
            index = instance.end
        if self.__document.text[index:sentence.end].strip():
            yield ((index, sentence.end), self.__document.text[index:sentence.end], "O")
            
    def get_segments(self):
        sentences = []
        for sentence in self.__document.sentences:
            segments = []
            for segment in self.__get_text_segment(sentence):
                segments.append(segment)
            sentences.append(segments)
        return sentences
        
