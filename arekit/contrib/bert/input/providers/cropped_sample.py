from arekit.common.data.input.providers.sample.cropped import CroppedSampleRowProvider
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider


class CroppedBertSampleRowProvider(CroppedSampleRowProvider):

    def __init__(self, text_b_template, text_terms_mapper, **kwargs):

        text_provider = BaseSingleTextProvider(text_terms_mapper=text_terms_mapper) \
            if text_b_template is None else PairTextProvider(text_b_prompt=text_b_template,
                                                             text_terms_mapper=text_terms_mapper)

        super(CroppedBertSampleRowProvider, self).__init__(text_provider=text_provider, **kwargs)
