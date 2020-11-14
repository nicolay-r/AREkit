import unittest

from arekit.common.entities.base import Entity
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news.base import RuAttitudesNews
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper


class TestPartOfSpeech(unittest.TestCase):

    def test_ruattitudes_news_text_parsing(self):
        news_it = RuAttitudesCollection.iter_news(version=RuAttitudesVersions.Debug,
                                                  get_news_index_func=lambda: 0)

        for news in news_it:
            assert(isinstance(news, RuAttitudesNews))
            for sentence in news.iter_sentences(return_text=False):
                parsed_text = news.EntitiesParser.parse(sentence)
                print type(parsed_text)
                self.__print_parsed_text(parsed_text)

    def test_rusentrel_news_text_parsing(self):
        stemmer = MystemWrapper()
        synonyms = RuSentRelSynonymsCollection.load_collection(stemmer=stemmer)
        news = RuSentRelNews.read_document(doc_id=1,
                                           synonyms=synonyms,
                                           version=RuSentRelVersions.V11)

        assert(isinstance(news, RuSentRelNews))
        first_sentence = news.get_sentence_by_index(9)
        parsed_text = news.EntitiesParser.parse(first_sentence)
        self.__print_parsed_text(parsed_text)

    def __print_parsed_text(self, parsed_text):
        assert(isinstance(parsed_text, list))
        print u"Length: {}".format(len(parsed_text))
        for t in parsed_text:
            if isinstance(t, Entity):
                print u"<{}>".format(t.Value),
            else:
                print u"'{}'".format(t),


if __name__ == '__main__':
    unittest.main()
