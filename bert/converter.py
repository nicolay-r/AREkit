from arekit.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.bert.formatters.sample.formats import SampleFormatters
from arekit.bert.result.base import BertResults
from arekit.bert.result.binary import BertBinaryResults
from arekit.bert.result.multiple import BertMultipleResults
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.opinions import compose_opinion_collection
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.collection import OpinionCollection

from arekit.bert.encoder import BertEncoder


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
            wrapped_linked_opinion_iter=linked_iter,
            labels_helper=labels_helper,
            label_calc_mode=label_calculation_mode)

        yield news_id, collection


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
