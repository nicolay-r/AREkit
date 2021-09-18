import logging
import unittest

from tests.processing.text.debug_text import debug_show_news_terms

from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.parser import TextParser

from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions
from arekit.contrib.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions


class TestTextParser(unittest.TestCase):

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
        print((type(frame_variants)))
        frame_variants.fill_from_iterable(variants_with_id=frames.iter_frame_id_and_variants(),
                                          overwrite_existed_variant=True,
                                          raise_error_on_existed_variant=False)

        # RuAttitudes options.
        options = RuSentRelNewsParseOptions(stemmer=stemmer,
                                            frame_variants_collection=frame_variants)

        # Reading synonyms collection.
        synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer)

        version = RuSentRelVersions.V11
        for doc_id in RuSentRelIOUtils.iter_collection_indices(version):

            # Parsing
            news = RuSentRelNews.read_document(doc_id=doc_id,
                                               synonyms=synonyms,
                                               version=version)

            # Perform text parsing.
            parsed_news = TextParser.parse_news(news, options)
            debug_show_news_terms(parsed_news=parsed_news)


if __name__ == '__main__':
    unittest.main()
