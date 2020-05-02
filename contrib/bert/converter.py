import numpy as np
import pandas as pd

from arekit.common.experiment.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import Label
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.contrib.bert.encoder import BertEncoder

from arekit.contrib.bert.formatters.opinions.base import BertOpinionsFormatter
from arekit.contrib.bert.formatters.result.binary import BertBinaryResults
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter
from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter


def iter_eval_collections(formatter_type,
                          experiment,
                          label_calculation_mode):

    assert(isinstance(formatter_type, unicode))
    assert(isinstance(label_calculation_mode, unicode))
    assert(isinstance(experiment, BaseExperiment))

    data_type = DataType.Test

    bert_results = BertBinaryResults.from_tsv(data_type=data_type, experiment=experiment)

    bert_test_samples = BertEncoder.create_formatter(data_type=data_type, formatter_type=formatter_type)
    bert_test_samples.from_tsv(experiment=experiment)

    bert_test_opinions = BertOpinionsFormatter(data_type=data_type)
    bert_test_opinions.from_tsv(experiment=experiment)

    print len(bert_test_samples)
    print len(bert_results)
    assert(len(bert_results) == len(bert_test_samples))

    collections = dict()

    # dict of news_id, opinion_id
    all_opinions = {}
    for row_index in range(len(bert_results)):

        print "Reading row: {} ({}%)".format(
            row_index,
            round(row_index * 100.0 / len(bert_results), 1))

        result_row = bert_results.iloc[row_index].tolist()

        # TODO. Now it assumes to perform a max (Need to support a binary format)
        label = Label.from_uint(np.argmax(result_row))

        samples_row = bert_test_samples.iloc[row_index].tolist()
        sample_row_id = BaseSampleFormatter.extract_row_id(samples_row)

        news_id = MultipleIDFormatter.parse_news_id(sample_row_id)

        opinion_id =MultipleIDFormatter.sample_row_id_to_opinion_id(sample_row_id)
        _, source, target = bert_test_opinions.parse_row(
            opinion_id=opinion_id)

        # TODO. Provide opinion iterator from classes in result/

        opinion = Opinion(source_value=source,
                          target_value=target,
                          sentiment=label)

        if news_id not in all_opinions:
            all_opinions[news_id] = {}

        news_opinions = all_opinions[news_id]
        if opinion_id not in news_opinions:
            news_opinions[opinion_id] = []

        news_opinions[opinion_id].append(opinion)

    for news_id, opinions_dict in all_opinions.iteritems():
        collections[news_id] = experiment.create_opinion_collection(
            opinions=list(__iter_opinions(opinions_dict=opinions_dict,
                                          label_calculation_mode=label_calculation_mode)))

    for news_id, collection in collections.iteritems():
        yield news_id, collection


def __iter_opinions(opinions_dict, label_calculation_mode):
    assert(isinstance(opinions_dict, dict))
    assert(isinstance(label_calculation_mode, unicode))

    for opinion_id, opinions in opinions_dict.iteritems():

        label = SingleLabelsHelper.create_label_from_text_opinions(
            text_opinion_labels=[o.Sentiment for o in opinions],
            label_creation_mode=label_calculation_mode)

        yield Opinion(source_value=opinions[0].SourceValue,
                      target_value=opinions[0].TargetValue,
                      sentiment=label)
