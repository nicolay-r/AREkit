import unittest

from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.pipeline.context import PipelineContext
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

        pipeline_ctx = PipelineContext(d={
            "sentence": BaseNewsSentence(self.text.split())
        })

        text_parser.run(pipeline_ctx)
        result = pipeline_ctx.provide("src")
        for t in result:
            print(t)


if __name__ == '__main__':
    unittest.main()
