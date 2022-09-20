from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader
from arekit.contrib.source.sentinerel.entities import SentiNerelEntityCollection
from arekit.contrib.source.sentinerel.io_utils import SentiNerelIOUtils, DEFAULT_VERSION


class SentiNerelDocReader(object):

    @staticmethod
    def read_text_relations(filename, version):
        assert(isinstance(filename, str))

        return SentiNerelIOUtils.read_from_zip(
            inner_path=SentiNerelIOUtils.get_annotation_innerpath(filename),
            process_func=lambda input_file: [
                relation for relation in BratAnnotationParser.parse_annotations(
                    input_file=input_file, encoding='utf-8-sig')["relations"]],
            version=version)

    @staticmethod
    def read_document(filename, doc_id, version=DEFAULT_VERSION, entities_to_ignore=None):
        assert(isinstance(filename, str))
        assert(isinstance(doc_id, int))

        def file_to_doc(input_file):
            sentences = BratDocumentSentencesReader.from_file(input_file=input_file, entities=entities)
            return BratNews(doc_id=doc_id, sentences=sentences, text_relations=text_relations)

        # TODO. #398 issue -- in some cases entities might be nested. Therefore we limit the set
        # TODO. of the potential named entities.
        eti = ["EFFECT_NEG", "EFFECT_POS", "ARGUMENT_NEG", "ARGUMENT_POS", "EVENT"] \
            if entities_to_ignore is None else entities_to_ignore

        entities = SentiNerelEntityCollection.read_collection(
            filename=filename, version=version, entities_to_ignore=eti)
        text_relations = SentiNerelDocReader.read_text_relations(filename=filename, version=version)

        return SentiNerelIOUtils.read_from_zip(
            inner_path=SentiNerelIOUtils.get_news_innerpath(filename=filename),
            process_func=file_to_doc,
            version=version)
