from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.contrib.experiments.common import entity_to_group_func
from arekit.contrib.bert.label.str_rus_fmt import RussianThreeScaleRussianLabelsFormatter
from arekit.contrib.bert.samplers.base import create_simple_sample_formatter
from arekit.contrib.bert.samplers.nli_b import NliBinarySampleFormatter
from arekit.contrib.bert.samplers.nli_m import NliMultipleSampleFormatter
from arekit.contrib.bert.samplers.qa_b import QaBinarySampleFormatter
from arekit.contrib.bert.samplers.qa_m import QaMultipleSampleFormatter
from arekit.contrib.bert.samplers.types import BertSampleFormatterTypes
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper


def create_bert_sample_formatter(data_type, formatter_type, label_scaler, balance,
                                 entity_formatter, entity_to_group_func):
    """
    This is a factory method, which allows to instantiate any of the
    supported bert_sample_encoders
    """
    assert(isinstance(formatter_type, BertSampleFormatterTypes))
    assert(callable(entity_to_group_func))
    assert(isinstance(entity_formatter, StringEntitiesFormatter))

    l_formatter = RussianThreeScaleRussianLabelsFormatter()
    text_terms_mapper = BertDefaultStringTextTermsMapper(
        entity_formatter=entity_formatter,
        entity_to_group_func=entity_to_group_func)

    if formatter_type == BertSampleFormatterTypes.CLASSIF_M:
        return create_simple_sample_formatter(data_type=data_type,
                                              label_scaler=label_scaler,
                                              text_terms_mapper=text_terms_mapper,
                                              balance=balance)
    if formatter_type == BertSampleFormatterTypes.NLI_M:
        return NliMultipleSampleFormatter(data_type=data_type,
                                          label_scaler=label_scaler,
                                          labels_formatter=l_formatter,
                                          text_terms_mapper=text_terms_mapper,
                                          balance=balance)
    if formatter_type == BertSampleFormatterTypes.QA_M:
        return QaMultipleSampleFormatter(data_type=data_type,
                                         label_scaler=label_scaler,
                                         labels_formatter=l_formatter,
                                         text_terms_mapper=text_terms_mapper,
                                         balance=balance)
    if formatter_type == BertSampleFormatterTypes.NLI_B:
        return NliBinarySampleFormatter(data_type=data_type,
                                        label_scaler=label_scaler,
                                        labels_formatter=l_formatter,
                                        text_terms_mapper=text_terms_mapper,
                                        balance=balance)
    if formatter_type == BertSampleFormatterTypes.QA_B:
        return QaBinarySampleFormatter(data_type=data_type,
                                       label_scaler=label_scaler,
                                       labels_formatter=l_formatter,
                                       text_terms_mapper=text_terms_mapper,
                                       balance=balance)

    return None
