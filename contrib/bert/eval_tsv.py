from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.model.labeling.base import LabelCalculationMode
from arekit.contrib.bert.converter import __iter_eval_collections

from arekit.contrib.bert.evaluation.opinion_based import BERTModelEvaluator
from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.base import BaseExperiment
from arekit.contrib.bert.formatters.sample.base import BaseSampleFormatter

from arekit.contrib.bert.formatters.opinions.base import OpinionsFormatter


def eval_tsv(data_io, data_type, experiment):
    assert(isinstance(data_io, DataIO))
    assert(isinstance(experiment, BaseExperiment))

    experiment.create_opinion_collection()

    bert_result_fp = u"test_results.tsv"

    opinions_fp = OpinionsFormatter.get_filepath(data_type=data_type,
                                                 experiment=experiment)

    samples_fp = BaseSampleFormatter.get_filepath(data_type=data_type,
                                                  experiment=experiment)

    iter_eval = __iter_eval_collections(bert_result_fp=bert_result_fp,
                                        experiment=experiment,
                                        samples_fp=samples_fp,
                                        opinion_fp=opinions_fp,
                                        label_calculation_mode=LabelCalculationMode.FIRST_APPEARED)

    for news_id, collection in iter_eval:

        filepath = experiment.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=news_id,
            epoch_index=0)

        experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                        filepath=filepath)

    bert_evaluator = BERTModelEvaluator(
        evaluator=TwoClassEvaluator(synonyms=data_io.SynonymsCollection),
        experiment=experiment)

    doc_ids = experiment.iter_news_indices(data_type=data_type)
    result = bert_evaluator.evaluate(data_type=data_type,
                                     doc_ids=experiment.iter_doc_ids_to_compare(doc_ids),
                                     epoch_index=0)

    assert(isinstance(result, BaseEvalResult))

    result.calculate()
    print result.get_result_as_str()
