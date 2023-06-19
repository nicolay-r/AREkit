from arekit.common.data.input.sample import InputSampleBase
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter


class DistanceLimitedTextOpinionFilter(TextOpinionFilter):

    def __init__(self, terms_per_context):
        super(DistanceLimitedTextOpinionFilter, self).__init__()
        self.__terms_per_context = terms_per_context

    def filter(self, text_opinion, parsed_doc, entity_service_provider):

        return InputSampleBase.check_ability_to_create_sample(
            entity_service=entity_service_provider,
            text_opinion=text_opinion,
            window_size=self.__terms_per_context)
