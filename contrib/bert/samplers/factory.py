from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.bert.entity.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.bert.label.str_rus_fmt import RussianThreeScaleRussianLabelsFormatter
from arekit.contrib.bert.samplers.base import create_simple_sample_formatter
from arekit.contrib.bert.samplers.nli_b import NliBinarySampleFormatter
from arekit.contrib.bert.samplers.nli_m import NliMultipleSampleFormatter
from arekit.contrib.bert.samplers.qa_b import QaBinarySampleFormatter
from arekit.contrib.bert.samplers.qa_m import QaMultipleSampleFormatter
from arekit.contrib.bert.samplers.types import BertSampleFormatterTypes
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper


def create_bert_sample_formatter(data_type, formatter_type, label_scaler,
                                 entity_formatter=None,
                                 synonyms=None):
    """
    This is a factory method, which allows to instantiate any of the
    supported bert_sample_encoders
    """
    assert(isinstance(formatter_type, BertSampleFormatterTypes))
    assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)
    assert(isinstance(entity_formatter, StringEntitiesFormatter))

    l_formatter = RussianThreeScaleRussianLabelsFormatter()
    e_formatter = RussianEntitiesFormatter() if entity_formatter is None else entity_formatter
    text_terms_mapper = BertDefaultStringTextTermsMapper(entity_formatter=e_formatter,
                                                         synonyms=synonyms)

    if formatter_type == BertSampleFormatterTypes.CLASSIF_M:
        return create_simple_sample_formatter(data_type=data_type,
                                              label_scaler=label_scaler,
                                              text_terms_mapper=text_terms_mapper)
    if formatter_type == BertSampleFormatterTypes.NLI_M:
        return NliMultipleSampleFormatter(data_type=data_type,
                                          label_scaler=label_scaler,
                                          labels_formatter=l_formatter,
                                          text_terms_mapper=text_terms_mapper)
    if formatter_type == BertSampleFormatterTypes.QA_M:
        return QaMultipleSampleFormatter(data_type=data_type,
                                         label_scaler=label_scaler,
                                         labels_formatter=l_formatter,
                                         text_terms_mapper=text_terms_mapper)
    if formatter_type == BertSampleFormatterTypes.NLI_B:
        return NliBinarySampleFormatter(data_type=data_type,
                                        label_scaler=label_scaler,
                                        labels_formatter=l_formatter,
                                        text_terms_mapper=text_terms_mapper)
    if formatter_type == BertSampleFormatterTypes.QA_B:
        return QaBinarySampleFormatter(data_type=data_type,
                                       label_scaler=label_scaler,
                                       labels_formatter=l_formatter,
                                       text_terms_mapper=text_terms_mapper)

    return None
