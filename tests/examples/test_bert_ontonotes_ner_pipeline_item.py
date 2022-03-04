import unittest

from arekit.common.news.base import News
from arekit.common.news.parser import NewsParser
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.text.parser import BaseTextParser
from arekit.processing.text.pipeline_terms_splitter import TermsSplitterParser
from examples.text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem


class BertOntonotesPipelineItemTest(unittest.TestCase):

    text = "США пытается ввести санкции против Российской Федерацией"

    def test_pipeline(self):
        text_parser = BaseTextParser([
            TermsSplitterParser(),
            BertOntonotesNERPipelineItem()
        ])
        news = News(doc_id=0, sentences=[BaseNewsSentence(self.text)])
        parsed_news = NewsParser.parse(news=news, text_parser=text_parser)
        terms = parsed_news.iter_sentence_terms(sentence_index=0, return_id=False)

        for term in terms:
            print(term)


if __name__ == '__main__':
    unittest.main()
