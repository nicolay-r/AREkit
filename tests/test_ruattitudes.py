#!/usr/bin/python2.7
import logging
import sys
import unittest

from arekit.common.entities.base import Entity
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.parser import TextParser
from arekit.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.source.ruattitudes.news.parse_options import RuAttitudesParseOptions

sys.path.append('../../')

from arekit.source.ruattitudes.news.base import RuAttitudesNews
from arekit.source.ruattitudes.collection import RuAttitudesCollection
from arekit.source.ruattitudes.sentence import RuAttitudesSentence

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class TestRuAttiudes(unittest.TestCase):

    def test_indices(self):
        ids = set()
        for news in RuAttitudesCollection.iter_news():
            assert(isinstance(news, RuAttitudesNews))
            assert(news.ID not in ids)
            ids.add(news.ID)

    def test_parsing(self):
        # Initializing stemmer
        stemmer = MystemWrapper()

        options = RuAttitudesParseOptions(stemmer=stemmer,
                                          frame_variants_collection=None)

        # iterating through collection
        for news in RuAttitudesCollection.iter_news():

            # parse news
            parsed_news = TextParser.parse_news(news=news, parse_options=options)
            terms = parsed_news.iter_sentence_terms(sentence_index=0,
                                                    return_id=False)

            str_terms = ['E' if isinstance(t, Entity) else t for t in terms]
            logger.info(" ".join(str_terms))

    def test_reading(self):

        # iterating through collection
        for news in RuAttitudesCollection.iter_news():
            assert(isinstance(news, RuAttitudesNews))
            logger.debug(u"News: {}".format(news.ID))

            for sentence in news.iter_sentences(return_text=False):
                assert(isinstance(sentence, RuAttitudesSentence))
                # text
                logger.debug(sentence.Text.encode('utf-8'))
                # objects
                logger.debug(u",".join([object.get_value() for object in sentence.iter_objects()]))
                # attitudes
                for ref_opinion in sentence.iter_ref_opinions():
                    src, target = sentence.get_objects(ref_opinion)
                    s = u"{src}->{target} ({label}) (t:[{src_type},{target_type}])".format(
                        src=src.get_value(),
                        target=target.get_value(),
                        label=str(ref_opinion.Sentiment.to_class_str()),
                        src_type=src.Type,
                        target_type=target.Type).encode('utf-8')
                    logger.debug(s)

            break


if __name__ == '__main__':
    unittest.main()
