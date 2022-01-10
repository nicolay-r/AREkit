from arekit.common.experiment.name_provider import ExperimentNameProvider


def create_infer_experiment_name_provider():
    return ExperimentNameProvider(name="example", suffix="infer")