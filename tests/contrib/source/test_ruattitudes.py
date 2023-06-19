import logging
import sys
import unittest
from collections import OrderedDict
from tqdm import tqdm

sys.path.append('../../../../')

from arekit.common.opinions.base import Opinion
from arekit.common.entities.base import Entity
from arekit.common.utils import progress_bar_iter
from arekit.common.docs.parser import DocumentParser
from arekit.common.text.parser import BaseTextParser
from arekit.common.context.token import Token
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler

from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.text_object import TextObject
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.doc import RuAttitudesDocument
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.ruattitudes.doc_brat import RuAttitudesDocumentsConverter

from tests.contrib.source.utils import RuAttitudesSentenceOpinionUtils
from tests.contrib.source.labels import PositiveLabel, NegativeLabel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class RuAttitudesLabelScaler(BaseLabelScaler):

    def __init__(self):

        self.__int_to_label_dict = OrderedDict([
            (self._neutral_label_instance(), 0),
            (self._positive_label_instance(), 1),
            (self._negative_label_instance(), -1)])

        self.__uint_to_label_dict = OrderedDict([
            (self._neutral_label_instance(), 0),
            (self._positive_label_instance(), 1),
            (self._negative_label_instance(), 2)])

        super(RuAttitudesLabelScaler, self).__init__(int_dict=self.__int_to_label_dict,
                                                     uint_dict=self.__uint_to_label_dict)

    @classmethod
    def _neutral_label_instance(cls):
        return NoLabel()

    @classmethod
    def _positive_label_instance(cls):
        return PositiveLabel()

    @classmethod
    def _negative_label_instance(cls):
        return NegativeLabel()


class TestRuAttitudes(unittest.TestCase):

    __ra_versions = [
        RuAttitudesVersions.V20Base,
        RuAttitudesVersions.V20Large,
        RuAttitudesVersions.V20BaseNeut,
        RuAttitudesVersions.V20LargeNeut,
    ]

    # region private methods

    def __check_entities(self, doc):
        for sentence in doc.iter_sentences():
            assert (isinstance(sentence, RuAttitudesSentence))
            for s_obj in sentence.iter_objects():
                assert (isinstance(s_obj, TextObject))
                entity = s_obj.to_entity(lambda in_id: in_id)
                assert (isinstance(entity, BratEntity))
                self.assertTrue(entity.GroupIndex is not None,
                                "Group index [{doc_id}] is None!".format(doc_id=doc.ID))

    def __iter_indices(self, ra_version):
        ids = set()
        for doc in tqdm(RuAttitudesCollection.iter_docs(version=ra_version,
                                                        get_doc_index_func=lambda _: len(ids),
                                                        return_inds_only=False)):
            assert(isinstance(doc, RuAttitudesDocument))
            assert(doc.ID not in ids)

            # Perform check for every entity.
            self.__check_entities(doc)

            ids.add(doc.ID)

    def __test_parsing(self, ra_version):
        # Initialize text parser pipeline.
        text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True)])

        # iterating through collection
        doc_read = 0

        doc_it = RuAttitudesCollection.iter_docs(version=ra_version,
                                                  get_doc_index_func=lambda _: doc_read,
                                                  return_inds_only=False)

        for doc in tqdm(doc_it):

            # parse doc
            brat_doc = RuAttitudesDocumentsConverter.to_brat_doc(doc)
            parsed_doc = DocumentParser.parse(doc=brat_doc, text_parser=text_parser)
            terms = parsed_doc.iter_sentence_terms(sentence_index=0,
                                                    return_id=False)

            str_terms = []
            for t in terms:
                if isinstance(t, Entity):
                    str_terms.append("E")
                elif isinstance(t, Token):
                    str_terms.append(t.get_token_value())
                else:
                    str_terms.append(t)

            for t in str_terms:
                self.assertIsInstance(t, str)

            doc_read += 1

    def __test_iter_doc_inds(self, ra_version):
        # iterating through collection
        doc_ids_it = RuAttitudesCollection.iter_docs(version=ra_version,
                                                     get_doc_index_func=lambda ind: ind + 1,
                                                     return_inds_only=True)

        it = progress_bar_iter(iterable=doc_ids_it,
                               desc="Extracting document ids",
                               unit="docs")

        print("Total documents count: {}".format(max(it)))

    def __test_reading(self, ra_version, do_printing=True):

        # iterating through collection
        doc_read = 0
        doc_it = RuAttitudesCollection.iter_docs(version=ra_version,
                                                  get_doc_index_func=lambda _: doc_read,
                                                  return_inds_only=False)

        if not do_printing:
            doc_it = tqdm(doc_it)

        for doc in doc_it:
            assert(isinstance(doc, RuAttitudesDocument))

            if not do_printing:
                continue

            logger.debug("Document: {}".format(doc.ID))

            label_scaler = RuAttitudesLabelScaler()

            for sentence in doc.iter_sentences():
                assert(isinstance(sentence, RuAttitudesSentence))
                # text
                logger.debug(sentence.Text)
                # objects
                logger.debug(",".join([object.Value for object in sentence.iter_objects()]))
                # attitudes
                for sentence_opin in sentence.iter_sentence_opins():
                    assert(isinstance(sentence_opin, SentenceOpinion))

                    source, target = sentence.get_objects(sentence_opin)
                    s = "{src}->{target} ({label}) (t:[{src_type},{target_type}]) tag=[{tag}]".format(
                        src=source.Value,
                        target=target.Value,
                        label=str(label_scaler.int_to_label(sentence_opin.Label)),
                        tag=sentence_opin.Tag,
                        src_type=str(source.Type),
                        target_type=str(target.Type))

                    logger.debug(sentence.SentenceIndex)
                    logger.debug(s)

                # Providing aggregated opinions.
                logger.info("Providing information for opinions with the related sentences:")
                data_it = RuAttitudesSentenceOpinionUtils.iter_opinions_with_related_sentences(
                    doc=doc, label_scaler=label_scaler)
                for o, sentences in data_it:
                    assert(isinstance(o, Opinion))
                    assert(isinstance(sentences, list))
                    logger.debug("'{source}'->'{target}' ({s_count})".format(
                        source=o.SourceValue,
                        target=o.TargetValue,
                        s_count=len(sentences)))

            doc_read += 1

    # endregion

    def test_indices(self):
        self.__iter_indices(ra_version=self.__ra_versions[2])

    def test_parsing(self):
        self.__test_parsing(ra_version=self.__ra_versions[2])

    def test_iter_doc_inds(self):
        self.__test_iter_doc_inds(ra_version=self.__ra_versions[2])

    def test_reading(self):
        self.__test_reading(ra_version=self.__ra_versions[2])

    def test_quick_reading_of_all_version(self):
        for version in self.__ra_versions:
            print("Testing version: {version}".format(version=version))
            self.__test_reading(ra_version=version, do_printing=False)


if __name__ == '__main__':
    unittest.main()
