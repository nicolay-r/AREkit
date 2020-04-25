import os
from arekit.contrib.experiments.base import BaseExperiment


def get_output_dir(data_type, experiment):
    assert(isinstance(experiment, BaseExperiment))
    filepath = experiment.create_result_opinion_collection_filepath(data_type=data_type,
                                                                    doc_id=0,
                                                                    epoch_index=0)
    return os.path.dirname(filepath)

