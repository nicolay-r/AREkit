from arekit.common.synonyms.base import SynonymsCollection
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader
from arekit.contrib.source.rusentrel.entities import RuSentRelDocumentEntityCollection
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils


class RuSentRelNewsReader(object):

    # region class methods

    @staticmethod
    def hide_first_entry(line, entry, hide_with=" "):

        index = line.find(entry)

        if index >= 0:
            pad = hide_with * len(entry)
            before = line[0:index]
            after = line[index+len(entry):]
            line = "".join([before, pad, after])

        return line

    @staticmethod
    def read_document(doc_id, synonyms, version=RuSentRelVersions.V11, target_doc_id=None):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(target_doc_id, int) or target_doc_id is None)

        def file_to_doc(input_file):

            sentences = BratDocumentSentencesReader.from_file(
                input_file=input_file,
                entities=entities,
                line_handler=lambda line: RuSentRelNewsReader.hide_first_entry(line, entry="{Author, Unknown}"),
                skip_entity_func=lambda entity: entity.Value in ['author', 'unknown'])

            return BratNews(doc_id=target_doc_id if target_doc_id is not None else doc_id,
                            sentences=sentences,
                            text_relations=[])

        entities = RuSentRelDocumentEntityCollection.read_collection(
            doc_id=doc_id,
            synonyms=synonyms,
            version=version)

        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_news_innerpath(index=doc_id, version=version),
            process_func=file_to_doc,
            version=version)
