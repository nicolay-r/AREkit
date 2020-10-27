from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.utils import create_dir_if_not_exists


# TODO. We save in file, which in general is incorrect.
def save_opinion_collections(opinion_collection_iter, experiment, data_type, labels_formatter, epoch_index):
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, DataType))
    assert(isinstance(labels_formatter, StringLabelsFormatter))
    assert(isinstance(epoch_index, int))

    for doc_id, collection in opinion_collection_iter:
        filepath = experiment.OpinionOperations.create_result_opinion_collection_filepath(
            data_type=data_type,
            doc_id=doc_id,
            epoch_index=epoch_index)

        create_dir_if_not_exists(filepath)

        experiment.DataIO.OpinionFormatter.save_to_file(collection=collection,
                                                        filepath=filepath,
                                                        labels_formatter=labels_formatter)
