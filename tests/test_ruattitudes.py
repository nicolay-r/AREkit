#!/usr/bin/python
import logging
import sys
import unittest

sys.path.append('../')

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

    def test_reading(self):
        # iterating through collection
        for news in RuAttitudesCollection.iter_news():
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


if __name__ == '__main__':
    unittest.main()
