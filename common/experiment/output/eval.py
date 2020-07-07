# TODO. Remove this dependency
# TODO. Remove this dependency
# TODO. Remove this dependency
from arekit.bert.evaluation.opinion_based import BERTModelEvaluator

from arekit.common.evaluation.evaluators.two_class import TwoClassEvaluator
from arekit.common.evaluation.results.base import BaseEvalResult
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.output.base import BaseOutput
from arekit.common.experiment.output.converter import OutputToOpinionCollectionsConverter


EPOCH_INDEX_PLACEHOLER = 0


def eval_output(samples_formatter_func,
                data_type,
                experiment,
                label_calculation_mode,
                output_instance):
    """
    Args:
        samples_formatter_func: func(data_type) -> FormatterType
    """
    assert(callable(samples_formatter_func))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(label_calculation_mode, unicode))
    assert(isinstance(output_instance, BaseOutput))

    experiment.OpinionOperations.create_opinion_collection()

    opinion_collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        samples_formatter_func=samples_formatter_func,
        experiment=experiment,
        label_calculation_mode=label_calculation_mode,
        output=output_instance)

    for news_id, collection in opinion_collections_iter:

        filepath = experiment.OpinionOperations.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=news_id,
            epoch_index=EPOCH_INDEX_PLACEHOLER)

        experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                        filepath=filepath)


# TODO. To experiment
# TODO. Need evaluator. (BaseModelEvaluator) + extra parameter.
def evaluate_bert_model(experiment, data_type):
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, DataType))

    bert_evaluator = BERTModelEvaluator(
        evaluator=TwoClassEvaluator(synonyms=experiment.DataIO.SynonymsCollection),
        opin_ops=experiment.OpinionOperations)

    doc_ids = experiment.DocumentOperations.iter_news_indices(data_type=data_type)
    result = bert_evaluator.evaluate(data_type=data_type,
                                     doc_ids=experiment.OpinionOperations.iter_doc_ids_to_compare(doc_ids),
                                     epoch_index=EPOCH_INDEX_PLACEHOLER)

    assert(isinstance(result, BaseEvalResult))
    return result.calculate()
