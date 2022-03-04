import unittest

from arekit.processing.entities.obj_desc import NerObjectDescriptor
from examples.text.ner_ontonotes import BertOntonotesNER


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


if __name__ == '__main__':
    unittest.main()
