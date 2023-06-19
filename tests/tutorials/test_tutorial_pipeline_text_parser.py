import logging
import unittest

from arekit.common.context.token import Token
from arekit.common.entities.base import Entity
from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import Label
from arekit.common.docs.base import Document
from arekit.common.docs.parser import DocumentParser
from arekit.common.docs.sentence import BaseDocumentSentence
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.pipelines.items.text.entities_default import TextEntitiesParser
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)


class PositiveTo(Label):
    pass


class NegativeTo(Label):
    pass


class TestTestParser(unittest.TestCase):

    def test(self):
        text = "А контроль над этими провинциями — [США] , которая не пытается ввести санкции против."

        frames_collection = RuSentiFramesCollection.read(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(
                pos_label_type=PositiveTo, neg_label_type=NegativeTo),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                pos_label_type=PositiveTo, neg_label_type=NegativeTo))

        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        text_parser = BaseTextParser(pipeline=[
            TextEntitiesParser(),
            DefaultTextTokenizer(keep_tokens=True),
            LemmasBasedFrameVariantsParser(frame_variants=frame_variant_collection,
                                           stemmer=MystemWrapper()),
            FrameVariantsSentimentNegation()
        ])

        doc = Document(doc_id=0, sentences=[BaseDocumentSentence(text.split())])
        parsed_doc = DocumentParser.parse(doc=doc, text_parser=text_parser)
        self.debug_show_terms(parsed_doc.iter_terms())

    @staticmethod
    def debug_show_terms(terms):
        for term in terms:
            if isinstance(term, str):
                logger.debug("Word:\t\t'{}'".format(term))
            elif isinstance(term, Token):
                logger.debug("Token:\t\t'{}' ('{}')".format(term.get_token_value(),
                                                            term.get_meta_value()))
            elif isinstance(term, Entity):
                logger.debug("Entity:\t\t'{}'".format(term.Value))
            elif isinstance(term, TextFrameVariant):
                text = "TextFV({is_neg}):\t'{v}'".format(is_neg="+" if term.IsNegated else "-",
                                                         v=term.Variant.get_value())
                logger.debug(text)
            else:
                raise Exception("unsuported type {}".format(term))


if __name__ == '__main__':
    unittest.main()
