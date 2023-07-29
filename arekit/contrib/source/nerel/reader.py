from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader
from arekit.contrib.source.nerel.entities import NerelEntityCollection
from arekit.contrib.source.nerel.io_utils import NerelIOUtils, DEFAULT_VERSION


class NerelDocReader(object):

    def __init__(self, version=DEFAULT_VERSION):
        self.__version = version
        self.__doc_fold = NerelIOUtils.map_doc_to_fold_type(version)

    def read_text_relations(self, filename):
        assert(isinstance(filename, str))

        return NerelIOUtils.read_from_zip(
            inner_path=NerelIOUtils.get_annotation_innerpath(
                folding_data_type=self.__doc_fold[filename],
                filename=filename),
            process_func=lambda input_file: [
                relation for relation in BratAnnotationParser.parse_annotations(
                    input_file=input_file, encoding='utf-8-sig')["relations"]],
            version=self.__version)

    def read_document(self, filename, doc_id, entities_to_ignore=None):
        assert(isinstance(filename, str))
        assert(isinstance(doc_id, int))

        def file_to_doc(input_file):
            sentences = BratDocumentSentencesReader.from_file(input_file=input_file, entities=entities)
            return BratNews(doc_id=doc_id, sentences=sentences, text_relations=text_relations)

        entities = NerelEntityCollection.read_collection(
            filename=filename, version=self.__version, entities_to_ignore=entities_to_ignore)

        text_relations = self.read_text_relations(filename=filename)

        return NerelIOUtils.read_from_zip(
            inner_path=NerelIOUtils.get_news_innerpath(
                folding_data_type=self.__doc_fold[filename], filename=filename),
            process_func=file_to_doc,
            version=self.__version)
