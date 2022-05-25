import logging
import unittest

from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.news.base import News
from arekit.common.news.parser import NewsParser
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.rusentrel.news_reader import RuSentRelNewsReader
from arekit.processing.text.pipeline_frames import FrameVariantsParser

from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_frames_negation import FrameVariantsSentimentNegation
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer
from arekit.processing.lemmatization.mystem import MystemWrapper

from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions

from tests.processing.text.debug_text import debug_show_news_terms


class TestTextParser(unittest.TestCase):

    def test_parse_single_string(self):
        text = "А контроль над этими провинциями — это господство над без малого половиной сирийской территории."
        parser = BaseTextParser(pipeline=[DefaultTextTokenizer(keep_tokens=True)])
        news = News(doc_id=0, sentences=[BaseNewsSentence(text.split())])
        parsed_news = NewsParser.parse(news=news, text_parser=parser)
        debug_show_news_terms(parsed_news=parsed_news)

    def test_parse_frame_variants(self):
        text = "США не пытается ввести санкции против Роccии"

        # Initializing stemmer.
        stemmer = MystemWrapper()

        # frame and variants.
        frames = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V20)
        frame_variants = FrameVariantsCollection()
        frame_variants.fill_from_iterable(variants_with_id=frames.iter_frame_id_and_variants(),
                                          overwrite_existed_variant=True,
                                          raise_error_on_existed_variant=False)

        parser = BaseTextParser(pipeline=[DefaultTextTokenizer(keep_tokens=True),
                                          FrameVariantsParser(frame_variants=frame_variants),
                                          LemmasBasedFrameVariantsParser(frame_variants=frame_variants,
                                                                         stemmer=stemmer),
                                          FrameVariantsSentimentNegation()])
        news = News(doc_id=0, sentences=[BaseNewsSentence(text.split())])
        parsed_news = NewsParser.parse(news=news, text_parser=parser)
        debug_show_news_terms(parsed_news=parsed_news)

    def test_parsing(self):

        # Initializing logger.
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)

        # Initializing stemmer.
        stemmer = MystemWrapper()

        # frame and variants.
        frames = RuSentiFramesCollection.read_collection(version=RuSentiFramesVersions.V20)
        frame_variants = FrameVariantsCollection()
        frame_variants.fill_from_iterable(variants_with_id=frames.iter_frame_id_and_variants(),
                                          overwrite_existed_variant=True,
                                          raise_error_on_existed_variant=False)

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True),
                                               LemmasBasedFrameVariantsParser(frame_variants=frame_variants,
                                                                              stemmer=stemmer,
                                                                              save_lemmas=False),
                                               FrameVariantsSentimentNegation()])

        # Reading synonyms collection.
        synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer)

        version = RuSentRelVersions.V11
        for doc_id in RuSentRelIOUtils.iter_collection_indices(version):

            # Parsing
            news = RuSentRelNewsReader.read_document(doc_id=doc_id,
                                                     synonyms=synonyms,
                                                     version=version)

            # Perform text parsing.
            parsed_news = NewsParser.parse(news=news, text_parser=text_parser)
            debug_show_news_terms(parsed_news=parsed_news)


if __name__ == '__main__':
    unittest.main()
