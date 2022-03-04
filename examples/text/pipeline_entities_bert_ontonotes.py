from arekit.common.bound import Bound
from arekit.common.entities.base import Entity
from arekit.common.news.objects_parser import SentenceObjectsParserPipelineItem
from arekit.common.text.partitioning.terms import TermsPartitioning
from arekit.processing.entities.obj_desc import NerObjectDescriptor
from examples.text.ner_ontonotes import BertOntonotesNER


class BertOntonotesNERPipelineItem(SentenceObjectsParserPipelineItem):

    def __init__(self):
        # Initialize bert-based model instance.
        self.__ontonotes_ner = BertOntonotesNER()
        super(BertOntonotesNERPipelineItem, self).__init__(TermsPartitioning())

    def _get_parts_provider_func(self, input_data, pipeline_ctx):
        return self.__iter_subs_values_with_bounds(input_data)

    def __iter_subs_values_with_bounds(self, terms_list):
        assert(isinstance(terms_list, list))

        single_sequence = [terms_list]
        processed_sequences = self.__ontonotes_ner.extract(sequences=single_sequence)

        for p_sequence in processed_sequences:
            for s_obj in p_sequence:
                assert(isinstance(s_obj, NerObjectDescriptor))
                value = " ".join(terms_list[s_obj.Position:s_obj.Position + s_obj.Length])
                entity = Entity(value=value, e_type=s_obj.ObjectType)
                yield entity, Bound(pos=s_obj.Position, length=s_obj.Length)
