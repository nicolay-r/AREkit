import unittest

from arekit.common.entities.base import Entity
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parser import NewsParser
from arekit.common.text.options import TextParseOptions
from arekit.common.text.parsed import BaseParsedText
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.experiment_rusentrel.labels.scalers.ruattitudes import ExperimentRuAttitudesLabelConverter
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.entities.parser import RuSentRelTextEntitiesParser
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.enums import TermFormat
from arekit.processing.text.token import Token

from arekit.processing.text.tokenizer import DefaultTextTokenizer


class TestPartOfSpeech(unittest.TestCase):

    def test_ruattitudes_news_text_parsing(self):
        news_it = RuAttitudesCollection.iter_news(version=RuAttitudesVersions.Debug,
                                                  get_news_index_func=lambda _: 0,
                                                  label_convereter=ExperimentRuAttitudesLabelConverter(),
                                                  return_inds_only=False)

        parse_options = TextParseOptions(stemmer=MystemWrapper(),
                                         frame_variants_collection=None)

        text_parser = BaseTextParser(parse_options=parse_options,
                                     pipeline=[RuAttitudesTextEntitiesParser()])

        for news in news_it:

            # Parse news via external parser.
            parsed_news = NewsParser.parse(news=news, text_parser=text_parser)
            assert(isinstance(parsed_news, ParsedNews))

            # Display result
            for parsed_text in parsed_news:
                self.__print_parsed_text(parsed_text)

    def test_rusentrel_news_text_parsing(self):
        stemmer = MystemWrapper()
        version = RuSentRelVersions.V11
        parse_options = TextParseOptions(stemmer=stemmer,
                                         frame_variants_collection=None)

        text_parser = BaseTextParser(parse_options=parse_options,
                                     pipeline=[RuSentRelTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True)
                                               ])

        synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer,
                                                                       version=version)
        news = RuSentRelNews.read_document(doc_id=1,
                                           synonyms=synonyms,
                                           version=version)

        # Parse news via external parser.
        parsed_news = NewsParser.parse(news=news, text_parser=text_parser)

        # Display result
        for parsed_text in parsed_news:
            self.__print_parsed_text(parsed_text)

        assert(isinstance(parsed_news, ParsedNews))

    @staticmethod
    def __print_parsed_text(parsed_text):
        assert(isinstance(parsed_text, BaseParsedText))

        terms_list = list(parsed_text.iter_terms(TermFormat.Raw))

        print("Length: {}".format(len(terms_list)))
        for t in terms_list:
            if isinstance(t, Entity):
                print("<{}>".format(t.Value), end=' ')
            elif isinstance(t, Token):
                print("[{}]".format(t.get_token_value()), end=' ')
            else:
                print("{{{}}}".format(t), end=' ')
        print()


if __name__ == '__main__':
    unittest.main()
