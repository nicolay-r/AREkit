from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader
from arekit.contrib.source.nerel.entities import NerelEntityCollection
from arekit.contrib.source.nerel.io_utils import NerelIOUtils, DEFAULT_VERSION


class NerelDocReader(object):

    @staticmethod
    def read_text_relations(folding_type, filename, version):
        assert(isinstance(filename, str))

        return NerelIOUtils.read_from_zip(
            inner_path=NerelIOUtils.get_annotation_innerpath(folding_data_type=folding_type, filename=filename),
            process_func=lambda input_file: [
                relation for relation in BratAnnotationParser.parse_annotations(
                    input_file=input_file, encoding='utf-8-sig')["relations"]],
            version=version)

    @staticmethod
    def read_document(filename, doc_id, doc_fold=None, version=DEFAULT_VERSION, entities_to_ignore=None):
        assert(isinstance(filename, str))
        assert(isinstance(doc_id, int))

        def file_to_doc(input_file):
            sentences = BratDocumentSentencesReader.from_file(input_file=input_file, entities=entities)
            return BratNews(doc_id=doc_id, sentences=sentences, text_relations=text_relations)

        entities = NerelEntityCollection.read_collection(
            filename=filename, version=version, entities_to_ignore=entities_to_ignore)

        doc_fold = NerelIOUtils.map_doc_to_fold_type(version) if doc_fold is None else doc_fold

        text_relations = NerelDocReader.read_text_relations(
            folding_type=doc_fold[filename], filename=filename, version=version)

        return NerelIOUtils.read_from_zip(
            inner_path=NerelIOUtils.get_news_innerpath(folding_data_type=doc_fold[filename], filename=filename),
            process_func=file_to_doc,
            version=version)
