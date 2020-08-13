#!/usr/bin/python2.7
import logging
import sys
import unittest

sys.path.append('../../')

from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.news.base import RuAttitudesNews
from arekit.contrib.source.ruattitudes.news.parse_options import RuAttitudesParseOptions
from arekit.contrib.source.ruattitudes.sentence.base import RuAttitudesSentence
from arekit.processing.text.token import Token

from arekit.common.entities.base import Entity
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.parser import TextParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestRuAttiudes(unittest.TestCase):

    def test_indices(self):
        ids = set()
        for news in RuAttitudesCollection.iter_news(version=RuAttitudesVersions.V11,
                                                    get_news_index_func=lambda: len(ids)):
            assert(isinstance(news, RuAttitudesNews))
            assert(news.ID not in ids)
            ids.add(news.ID)

    def test_parsing(self):
        # Initializing stemmer
        stemmer = MystemWrapper()

        options = RuAttitudesParseOptions(stemmer=stemmer,
                                          frame_variants_collection=None)

        # iterating through collection
        news_readed = 0

        news_it = RuAttitudesCollection.iter_news(version=RuAttitudesVersions.V11,
                                                  get_news_index_func=lambda: news_readed)

        for news in news_it:

            # parse news
            parsed_news = TextParser.parse_news(news=news, parse_options=options)
            terms = parsed_news.iter_sentence_terms(sentence_index=0,
                                                    return_id=False)

            str_terms = []
            for t in terms:
                if isinstance(t, Entity):
                    str_terms.append(u"E")
                elif isinstance(t, Token):
                    str_terms.append(t.get_token_value())
                else:
                    str_terms.append(t)

            for t in str_terms:
                self.assertIsInstance(t, unicode)

            logger.info(u" ".join(str_terms))
            news_readed += 1

    def test_reading(self):

        # iterating through collection
        news_readed = 0
        news_it = RuAttitudesCollection.iter_news(version=RuAttitudesVersions.V11,
                                                  get_news_index_func=lambda: news_readed)
        for news in news_it:
            assert(isinstance(news, RuAttitudesNews))
            logger.debug(u"News: {}".format(news.ID))

            for sentence in news.iter_sentences(return_text=False):
                assert(isinstance(sentence, RuAttitudesSentence))
                # text
                logger.debug(sentence.Text.encode('utf-8'))
                # objects
                logger.debug(u",".join([object.get_value() for object in sentence.iter_objects()]))
                # attitudes
                for sentence_opin in sentence.iter_sentence_opins():
                    src, target = sentence.get_objects(sentence_opin)
                    s = u"{src}->{target} ({label}) (t:[{src_type},{target_type}])".format(
                        src=src.get_value(),
                        target=target.get_value(),
                        label=str(sentence_opin.Sentiment.to_class_str()),
                        src_type=src.Type,
                        target_type=target.Type).encode('utf-8')
                    logger.debug(s)

            news_readed += 1
            break


if __name__ == '__main__':
    unittest.main()