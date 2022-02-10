import unittest

from arekit.common.news.base import News
from arekit.common.news.parser import NewsParser
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.text.parser import BaseTextParser
from arekit.processing.entities.bert_ontonotes import BertOntonotesNER
from arekit.processing.entities.obj_desc import NerObjectDescriptor
from arekit.processing.text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem


class BertOntonotesTest(unittest.TestCase):

    text = ".. При этом Москва неоднократно подчеркивала, что ее активность " \
             "на балтике является ответом именно на действия НАТО и эскалацию " \
             "враждебного подхода к Росcии вблизи ее восточных границ ..."

    def test_single_inference(self):
        ner = BertOntonotesNER()
        tokens = self.text.split(' ')
        sequences = ner.extract(sequences=[tokens])

        print(len(sequences))
        for s_objs in sequences:
            for s_obj in s_objs:
                assert (isinstance(s_obj, NerObjectDescriptor))
                print("----")
                print(s_obj.ObjectType)
                print(s_obj.Position)
                print(s_obj.Length)
                print(tokens[s_obj.Position:s_obj.Position + s_obj.Length])

    def test_pipeline(self):
        text_parser = BaseTextParser([BertOntonotesNERPipelineItem()])
        news = News(doc_id=0, sentences=[BaseNewsSentence(self.text.split())])
        parsed_news = NewsParser.parse(news=news, text_parser=text_parser)
        print(parsed_news.iter_sentence_terms(sentence_index=0))


if __name__ == '__main__':
    unittest.main()
