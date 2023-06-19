import unittest
from enum import Enum
from os.path import basename, join, dirname

from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.entities.compound import BratCompoundEntity
from arekit.contrib.source.brat.news import BratDocument
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader
from arekit.contrib.source.zip_utils import ZipArchiveUtils
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


class FooVersions(Enum):
    V1 = "V1"


class FooIOUtils(ZipArchiveUtils):

    archive_path = join(dirname(__file__), "data/foo.zip")

    @staticmethod
    def get_archive_filepath(version):
        return FooIOUtils.archive_path

    @staticmethod
    def get_annotation_innerpath(filename):
        return "{}.ann".format(filename)

    @staticmethod
    def get_news_innerpath(filename):
        return "{}.txt".format(filename)

    @staticmethod
    def __iter_filenames_from_dataset():
        for filename in FooIOUtils.iter_filenames_from_zip(FooVersions.V1):
            yield basename(filename)

    @staticmethod
    def iter_collection_filenames():
        filenames_it = FooIOUtils.__iter_filenames_from_dataset()
        for doc_id, filename in enumerate(filenames_it):
            yield doc_id, filename


class FooEntityCollection(EntityCollection):

    def __init__(self, contents, value_to_group_id_func):
        super(FooEntityCollection, self).__init__(contents["entities"], value_to_group_id_func)
        self._sort_entities(key=lambda entity: entity.IndexBegin)

    @classmethod
    def read_collection(cls, filename, version):
        synonyms = StemmerBasedSynonymCollection(stemmer=MystemWrapper(), is_read_only=False)
        return FooIOUtils.read_from_zip(
            inner_path=FooIOUtils.get_annotation_innerpath(filename),
            process_func=lambda input_file: cls(
                contents=BratAnnotationParser.parse_annotations(input_file),
                value_to_group_id_func=lambda value:
                    SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                        synonyms, value)),
            version=version)


class FooDocReader(object):

    @staticmethod
    def read_text_relations(filename, version):
        return FooIOUtils.read_from_zip(
            inner_path=FooIOUtils.get_annotation_innerpath(filename),
            process_func=lambda input_file: [relation for relation in
                                             BratAnnotationParser.parse_annotations(input_file)["relations"]],
            version=version)

    @staticmethod
    def read_document(filename, doc_id, version=FooVersions.V1):
        def file_to_doc(input_file):
            sentences = BratDocumentSentencesReader.from_file(input_file, entities)
            return BratDocument(doc_id, sentences, text_relations)

        entities = FooEntityCollection.read_collection(filename, version)
        text_relations = FooDocReader.read_text_relations(filename, version)

        return FooIOUtils.read_from_zip(
            inner_path=FooIOUtils.get_news_innerpath(filename),
            process_func=file_to_doc,
            version=version)


class TestFooCollection(unittest.TestCase):

    def test_reading(self):
        news = FooDocReader.read_document("0", doc_id=0)
        for sentence in news.iter_sentences():
            print(sentence.Text.strip())
            for entity, bound in sentence.iter_entity_with_local_bounds():
                print("{}: ['{}',{}, {}] {}".format(
                    entity.ID, entity.Value, entity.Type,
                    "-".join([str(bound.Position), str(bound.Position + bound.Length)]),
                    "[COMPOUND]" if isinstance(entity, BratCompoundEntity) else ""))

                if not isinstance(entity, BratCompoundEntity):
                    continue

                for child in entity.iter_childs():
                    print("\t{}: ['{}',{}]".format(child.ID, child.Value, child.Type))

        for brat_relation in news.Relations:
            print(brat_relation.SourceID, brat_relation.TargetID, brat_relation.Type)


if __name__ == '__main__':
    unittest.main()
