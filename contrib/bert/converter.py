import collections

from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.opinions import compose_opinion_collection
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.model.labeling.base import LabelsHelper
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

    bert_test_samples = BertEncoder.create_formatter(data_type=data_type,
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

    labels_helper = SingleLabelsHelper(label_scaler=experiment.DataIO.LabelsScaler)

    for news_id in bert_results.iter_news_ids():

        collection = experiment.OpinionOperations.create_opinion_collection()
        assert(isinstance(collection, OpinionCollection))

        linked_iter = bert_results.iter_linked_opinions(news_id=news_id,
                                                        bert_opinions=bert_test_opinions)

        collection = compose_opinion_collection(
            create_collection_func=experiment.OpinionOperations.create_opinion_collection,
            opinions_iter=__iter_opinions(linked_iter=linked_iter,
                                          labels_helper=labels_helper,
                                          label_calculation_mode=label_calculation_mode))

        yield news_id, collection


def __iter_opinions(linked_iter, label_calculation_mode, labels_helper):
    assert(isinstance(linked_iter, collections.Iterable))
    assert(isinstance(labels_helper, LabelsHelper))
    assert(isinstance(label_calculation_mode, unicode))

    for linked_opinions in linked_iter:
        yield __to_doc_opinion(linked_wrap=linked_opinions,
                               labels_helper=labels_helper,
                               label_calculation_mode=label_calculation_mode)


def __to_doc_opinion(linked_wrap, labels_helper, label_calculation_mode):
    assert(isinstance(linked_wrap, LinkedDataWrapper))
    assert(isinstance(labels_helper, LabelsHelper))
    assert(isinstance(label_calculation_mode, unicode))

    label = labels_helper.aggregate_labels(
        labels_list=[opinion.Sentiment for opinion in linked_wrap],
        label_creation_mode=label_calculation_mode)

    return Opinion(source_value=linked_wrap[0].SourceValue,
                   target_value=linked_wrap[0].TargetValue,
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

    if SampleFormatters.is_multiple(formatter_type):
        results = BertMultipleResults(labels_scaler=labels_scaler)
        results.from_tsv(data_type=data_type, experiment=experiment, ids_values=ids_values)

    return results
