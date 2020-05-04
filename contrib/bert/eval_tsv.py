from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.experiment.base import BaseExperiment

from arekit.contrib.bert.converter import iter_eval_collections
from arekit.contrib.bert.evaluation.opinion_based import BERTModelEvaluator


EPOCH_INDEX_PLACEHOLER = 0


def eval_tsv(formatter_type, data_type, experiment, label_calculation_mode):
    assert(isinstance(formatter_type, unicode))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(label_calculation_mode, unicode))

    experiment.create_opinion_collection()

    iter_eval = iter_eval_collections(formatter_type=formatter_type,
                                      experiment=experiment,
                                      label_calculation_mode=label_calculation_mode)

    for news_id, collection in iter_eval:

        filepath = experiment.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=news_id,
            epoch_index=EPOCH_INDEX_PLACEHOLER)

        experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                        filepath=filepath)


def evaluate_bert_model(experiment, data_type):
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, unicode))

    bert_evaluator = BERTModelEvaluator(
        evaluator=TwoClassEvaluator(synonyms=experiment.DataIO.SynonymsCollection),
        experiment=experiment)

    doc_ids = experiment.iter_news_indices(data_type=data_type)
    result = bert_evaluator.evaluate(data_type=data_type,
                                     doc_ids=experiment.iter_doc_ids_to_compare(doc_ids),
                                     epoch_index=EPOCH_INDEX_PLACEHOLER)

    assert(isinstance(result, BaseEvalResult))
    return result.calculate()
