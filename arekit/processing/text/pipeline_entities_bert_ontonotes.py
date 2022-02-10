from arekit.common.bound import Bound
from arekit.common.entities.base import Entity
from arekit.common.news.objects_parser import SentenceObjectsParserPipelineItem
from arekit.common.news.sentence import BaseNewsSentence
from arekit.processing.entities.bert_ontonotes import BertOntonotesNER
from arekit.processing.entities.obj_desc import NerObjectDescriptor


class BertOntonotesNERPipelineItem(SentenceObjectsParserPipelineItem):

    def __init__(self):
        # Initialize bert-based model instance.
        self.__ontonotes_ner = BertOntonotesNER()
        super(BertOntonotesNERPipelineItem, self).__init__(
            iter_objs_func=self.__iter_subs_values_with_bounds)

    def __iter_subs_values_with_bounds(self, sentence):
        """ Considering list of terms.
        """
        assert(isinstance(sentence, BaseNewsSentence))
        original_text = sentence.Text
        original_terms = original_text if isinstance(original_text, list) else original_text.split(' ')
        single_sequence = [original_terms]
        processed_sequences = self.__ontonotes_ner.extract(sequences=single_sequence)

        id_in_doc = 0

        for p_sequence in processed_sequences:
            for s_obj in p_sequence:
                assert (isinstance(s_obj, NerObjectDescriptor))

                value = " ".join(original_terms[s_obj.Position:s_obj.Position + s_obj.Length])

                entity = Entity(value=value,
                                e_type=s_obj.ObjectType,
                                id_in_doc=id_in_doc)

                yield entity, Bound(pos=s_obj.Position, length=s_obj.Length)

                id_in_doc += 1
