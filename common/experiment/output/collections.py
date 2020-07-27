from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment


def save_collections(opinion_collection_iter, experiment, data_type):
    """
        opinion_collection_iter: iter
            iter pairs (news_id, collection)
        experiment: BaseExperiment
        data_type: DataType
    """
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, DataType))

    for news_id, collection in opinion_collection_iter:
        filepath = experiment.OpinionOperations.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=news_id,
            epoch_index=experiment.EPOCH_INDEX_PLACEHOLER)

        experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                        filepath=filepath)

