from arekit.common.experiment.formats.base import BaseExperiment


def get_output_dir(experiment):
    assert(isinstance(experiment, BaseExperiment))
    return experiment.DataIO.get_model_root()


def generate_filename(data_type, experiment, prefix):
    assert(isinstance(data_type, unicode))
    assert(isinstance(prefix, unicode))
    assert(isinstance(experiment, BaseExperiment))

    return u"{prefix}-{data_type}-{cv_index}.csv".format(
        prefix=prefix,
        data_type=data_type,
        cv_index=experiment.DataIO.CVFoldingAlgorithm.IterationIndex)
