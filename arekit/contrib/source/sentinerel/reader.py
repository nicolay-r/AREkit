from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader
from arekit.contrib.source.sentinerel.entities import SentiNerelEntityCollection
from arekit.contrib.source.sentinerel.io_utils import SentiNerelIOUtils, SentiNerelVersions


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
    def read_document(filename, doc_id):
        assert(isinstance(filename, str))
        assert(isinstance(doc_id, int))

        def file_to_doc(input_file):
            sentences = BratDocumentSentencesReader.from_file(input_file=input_file, entities=entities)
            return BratNews(doc_id=doc_id, sentences=sentences, text_relations=text_relations)

        entities = SentiNerelEntityCollection.read_collection(filename=filename, version=SentiNerelVersions.V1)
        text_relations = SentiNerelDocReader.read_text_relations(filename=filename, version=SentiNerelVersions.V1)

        return SentiNerelIOUtils.read_from_zip(
            inner_path=SentiNerelIOUtils.get_news_innerpath(filename=filename),
            process_func=file_to_doc,
            version=SentiNerelVersions.V1)
