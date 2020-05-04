from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection

from arekit.contrib.bert.encoder import BertEncoder
from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.contrib.bert.formatters.result.base import BertResults
from arekit.contrib.bert.formatters.result.binary import BertBinaryResults
from arekit.contrib.bert.formatters.result.multiple import BertMultipleResults
from arekit.contrib.bert.formatters.sample.formats import SampleFormatters


def iter_eval_collections(formatter_type,
                          experiment,
                          label_calculation_mode):

    assert(isinstance(formatter_type, unicode))
    assert(isinstance(label_calculation_mode, unicode))
    assert(isinstance(experiment, BaseExperiment))

    data_type = DataType.Test

    bert_test_samples = BertEncoder.create_formatter(
        data_type=data_type,
        formatter_type=formatter_type,
        label_scaler=experiment.DataIO.LabelsScaler)

    bert_test_samples.from_tsv(experiment=experiment)

    bert_results = __read_results(formatter_type=formatter_type,
                                  data_type=data_type,
                                  experiment=experiment,
                                  ids_values=bert_test_samples.extract_ids(),
                                  labels_scaler=experiment.DataIO.LabelsScaler)

    assert(isinstance(bert_results, BertResults))

    bert_test_opinions = BertOpinionsFormatter(data_type=data_type)
    bert_test_opinions.from_tsv(experiment=experiment)

    assert(len(bert_results) == len(bert_test_samples))

    for news_id in bert_results.iter_news_ids():

        collection = experiment.create_opinion_collection()
        assert(isinstance(collection, OpinionCollection))

        linked_iter = bert_results.iter_wrapped_linked_text_opinions(news_id=news_id,
                                                                     bert_opinions=bert_test_opinions)

        for linked_wrap in linked_iter:
            opinion = __to_doc_opinion(linked_wrap=linked_wrap,
                                       label_calculation_mode=label_calculation_mode)
            collection.add_opinion(opinion)

        yield news_id, collection


def __to_doc_opinion(linked_wrap, label_calculation_mode):
    assert(isinstance(linked_wrap, LinkedTextOpinionsWrapper))
    assert(isinstance(label_calculation_mode, unicode))

    label = SingleLabelsHelper.create_label_from_text_opinions(
        text_opinion_labels=[opinion.Sentiment for opinion in linked_wrap],
        label_creation_mode=label_calculation_mode)

    return Opinion(source_value=linked_wrap.FirstOpinion.SourceValue,
                   target_value=linked_wrap.FirstOpinion.TargetValue,
                   sentiment=label)


def __read_results(formatter_type, data_type, experiment, ids_values, labels_scaler):
    assert(isinstance(formatter_type, unicode))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, unicode))
    assert(isinstance(ids_values, list))
    assert(isinstance(labels_scaler, BaseLabelScaler))

    results = None

    if SampleFormatters.is_binary(formatter_type):
        results = BertBinaryResults(labels_scaler=labels_scaler)
        results.from_tsv(data_type=data_type, experiment=experiment, ids_values=ids_values)

    if SampleFormatters.is_binary(formatter_type):
        results = BertMultipleResults(labels_scaler=labels_scaler)
        results.from_tsv(data_type=data_type, experiment=experiment, ids_values=ids_values)

    return results
