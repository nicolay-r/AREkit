from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.input.storages.sample import BaseSampleStorage
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.samplers.base import create_simple_sample_provider
from arekit.contrib.bert.samplers.nli_b import NliBinarySampleProvider
from arekit.contrib.bert.samplers.nli_m import NliMultipleSampleProvider
from arekit.contrib.bert.samplers.qa_b import QaBinarySampleProvider
from arekit.contrib.bert.samplers.qa_m import QaMultipleSampleProvider
from arekit.contrib.bert.samplers.types import BertSampleProviderTypes
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper


def create_bert_sample_provider(storage, provider_type, label_scaler,
                                labels_formatter, entity_formatter, entity_to_group_func):
    """
    This is a factory method, which allows to instantiate any of the
    supported bert_sample_encoders
    """
    assert(isinstance(storage, BaseSampleStorage))
    assert(isinstance(provider_type, BertSampleProviderTypes))
    assert(callable(entity_to_group_func))
    assert(isinstance(entity_formatter, StringEntitiesFormatter))
    assert(isinstance(labels_formatter, StringLabelsFormatter))

    text_terms_mapper = BertDefaultStringTextTermsMapper(
        entity_formatter=entity_formatter,
        entity_to_group_func=entity_to_group_func)

    if provider_type == BertSampleProviderTypes.CLASSIF_M:
        return create_simple_sample_provider(storage=storage,
                                             label_scaler=label_scaler,
                                             text_terms_mapper=text_terms_mapper)
    if provider_type == BertSampleProviderTypes.NLI_M:
        return NliMultipleSampleProvider(storage=storage,
                                         label_scaler=label_scaler,
                                         labels_formatter=labels_formatter,
                                         text_terms_mapper=text_terms_mapper)
    if provider_type == BertSampleProviderTypes.QA_M:
        return QaMultipleSampleProvider(storage=storage,
                                        label_scaler=label_scaler,
                                        labels_formatter=labels_formatter,
                                        text_terms_mapper=text_terms_mapper)

    if provider_type == BertSampleProviderTypes.NLI_B:
        return NliBinarySampleProvider(storage=storage,
                                       label_scaler=label_scaler,
                                       labels_formatter=labels_formatter,
                                       text_terms_mapper=text_terms_mapper)
    if provider_type == BertSampleProviderTypes.QA_B:
        return QaBinarySampleProvider(storage=storage,
                                      label_scaler=label_scaler,
                                      labels_formatter=labels_formatter,
                                      text_terms_mapper=text_terms_mapper)

    return None
