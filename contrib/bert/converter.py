import numpy as np
import pandas as pd

from arekit.common.experiment.base import BaseExperiment
from arekit.common.labels.base import Label
from arekit.common.model.labeling.base import LabelCalculationMode
from arekit.common.opinions.base import Opinion
from arekit.contrib.bert.formatters.opinion import OpinionsFormatter
from arekit.contrib.bert.formatters.row_ids.multiple import MultipleIDFormatter
from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter


def __iter_eval_collections(bert_result_fp,
                            samples_fp,
                            opinion_fp,
                            experiment,
                            label_calculation_mode,
                            classes_count=3):
    assert(isinstance(bert_result_fp, unicode))
    assert(isinstance(samples_fp, unicode))
    assert(isinstance(opinion_fp, unicode))
    assert(isinstance(label_calculation_mode, unicode))
    assert(isinstance(experiment, BaseExperiment))

    print "bert_output: {}".format(bert_result_fp)
    print "samples: {}".format(samples_fp)
    print "opinions: {}".format(opinion_fp)

    # bert results: p_neut, p_pos, p_neg
    df_bert_results = pd.read_csv(bert_result_fp, sep='\t', header=None)

    # origin: [id, ...]
    df_samples = pd.read_csv(samples_fp, sep='\t')

    # df opinions: [subj, obj, news_id]
    df_opinions = pd.read_csv(opinion_fp, sep='\t', header=None)
    df_opinions.columns = ['id', 'source', 'target']

    print len(df_samples)
    print len(df_bert_results)
    assert(len(df_bert_results) == len(df_samples))

    collections = dict()

    # dict of news_id, opinion_id
    all_opinions = {}
    for row_index in range(len(df_bert_results)):

        print "Reading row: {} ({}%)".format(
            row_index,
            round(row_index * 100.0 / len(df_bert_results), 1))

        result_row = df_bert_results.iloc[row_index].tolist()
        assert(len(result_row) == classes_count)

        # TODO. Now it assumes to perform a max (Need to support a binary format)
        label = Label.from_uint(np.argmax(result_row))

        samples_row = df_samples.iloc[row_index].tolist()
        sample_row_id = BaseSampleFormatter.extract_row_id(samples_row)

        news_id = MultipleIDFormatter.parse_news_id(sample_row_id)
        opinion_id = MultipleIDFormatter.sample_row_id_to_opinion_id(sample_row_id)

        _opinion_row = df_opinions[df_opinions['id'] == opinion_id]
        opinion_row = _opinion_row.iloc[0].tolist()

        _, source, target = OpinionsFormatter.parse_row(opinion_row)

        opinion = Opinion(source_value=source,
                          target_value=target,
                          sentiment=label)

        if news_id not in all_opinions:
            all_opinions[news_id] = {}

        news_opininons = all_opinions[news_id]
        if opinion_id not in news_opininons:
            news_opininons[opinion_id] = []

        news_opininons[opinion_id].append(opinion)

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
        # Recalculate average label

        label = __calculate_label(lcm=label_calculation_mode,
                                  labels=[o.Sentiment for o in opinions])

        yield Opinion(source_value=opinions[0].SourceValue,
                      target_value=opinions[0].TargetValue,
                      sentiment=label)


def __calculate_label(lcm, labels):
    assert(isinstance(lcm, unicode))
    assert(isinstance(labels, list))

    if lcm == LabelCalculationMode.AVERAGE:
        avg_label = np.sign(sum([l.to_int() for l in labels]))
        return Label.from_int(avg_label)

    if lcm == LabelCalculationMode.FIRST_APPEARED:
        return labels[0]