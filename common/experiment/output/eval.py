from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.output.base import BaseOutput
from arekit.common.experiment.output.converter import OutputToOpinionCollectionsConverter


def eval_output(
        # TODO. BaseInputReader (sample)
        # TODO. BaseInputReader (opinion)
        samples_formatter_func,
        data_type,
        experiment,
        label_calculation_mode,
        output_instance):
    """
    Args:
        samples_formatter_func: func(data_type) -> FormatterType
    """
    # TODO. Remove (as we utilize only reader instead).
    assert(callable(samples_formatter_func))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(label_calculation_mode, unicode))
    assert(isinstance(output_instance, BaseOutput))

    experiment.OpinionOperations.create_opinion_collection()

    opinion_collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        # TODO. BaseInputReader (sample)
        # TODO. BaseInputReader (opinion)
        samples_formatter_func=samples_formatter_func,
        experiment=experiment,
        label_calculation_mode=label_calculation_mode,
        output=output_instance)

    for news_id, collection in opinion_collections_iter:

        filepath = experiment.OpinionOperations.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=news_id,
            epoch_index=experiment.EPOCH_INDEX_PLACEHOLER)

        experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                        filepath=filepath)

