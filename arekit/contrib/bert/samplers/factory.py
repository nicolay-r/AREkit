from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.samplers.base import create_simple_sample_provider
from arekit.contrib.bert.samplers.nli_b import NliBinarySampleProvider
from arekit.contrib.bert.samplers.nli_m import NliMultipleSampleProvider
from arekit.contrib.bert.samplers.qa_b import QaBinarySampleProvider
from arekit.contrib.bert.samplers.qa_m import QaMultipleSampleProvider
from arekit.contrib.bert.samplers.types import BertSampleProviderTypes
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper


def create_bert_sample_provider(provider_type, label_scaler,
                                text_b_labels_fmt, entity_formatter):
    """
    This is a factory method, which allows to instantiate any of the
    supported bert_sample_encoders
    """
    assert(isinstance(provider_type, BertSampleProviderTypes))
    assert(isinstance(entity_formatter, StringEntitiesFormatter))
    assert(isinstance(text_b_labels_fmt, StringLabelsFormatter))

    text_terms_mapper = BertDefaultStringTextTermsMapper(entity_formatter)

    if provider_type == BertSampleProviderTypes.CLASSIF_M:
        return create_simple_sample_provider(label_scaler=label_scaler,
                                             text_terms_mapper=text_terms_mapper)
    if provider_type == BertSampleProviderTypes.NLI_M:
        return NliMultipleSampleProvider(label_scaler=label_scaler,
                                         text_b_labels_fmt=text_b_labels_fmt,
                                         text_terms_mapper=text_terms_mapper)
    if provider_type == BertSampleProviderTypes.QA_M:
        return QaMultipleSampleProvider(label_scaler=label_scaler,
                                        text_b_labels_fmt=text_b_labels_fmt,
                                        text_terms_mapper=text_terms_mapper)

    if provider_type == BertSampleProviderTypes.NLI_B:
        return NliBinarySampleProvider(label_scaler=label_scaler,
                                       text_b_labels_fmt=text_b_labels_fmt,
                                       text_terms_mapper=text_terms_mapper)
    if provider_type == BertSampleProviderTypes.QA_B:
        return QaBinarySampleProvider(label_scaler=label_scaler,
                                      text_b_labels_fmt=text_b_labels_fmt,
                                      text_terms_mapper=text_terms_mapper)

    return None
