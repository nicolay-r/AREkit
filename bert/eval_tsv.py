from arekit.bert.evaluation.opinion_based import BERTModelEvaluator
from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.experiment.formats.base import BaseExperiment

from arekit.bert.converter import iter_eval_collections


EPOCH_INDEX_PLACEHOLER = 0


def eval_tsv(formatter_type, data_type, experiment, label_calculation_mode):
    assert(isinstance(formatter_type, unicode))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(label_calculation_mode, unicode))

    experiment.OpinionOperations.create_opinion_collection()

    iter_eval = iter_eval_collections(formatter_type=formatter_type,
                                      experiment=experiment,
                                      label_calculation_mode=label_calculation_mode)

    for news_id, collection in iter_eval:

        filepath = experiment.OpinionOperations.create_result_opinion_collection_filepath(
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
        opin_ops=experiment.OpinionOperations)

    doc_ids = experiment.DocumentOperations.iter_news_indices(data_type=data_type)
    result = bert_evaluator.evaluate(data_type=data_type,
                                     doc_ids=experiment.OpinionOperations.iter_doc_ids_to_compare(doc_ids),
                                     epoch_index=EPOCH_INDEX_PLACEHOLER)

    assert(isinstance(result, BaseEvalResult))
    return result.calculate()
