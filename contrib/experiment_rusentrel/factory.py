from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.ds.experiment import RuAttitudesExperiment
from arekit.contrib.experiment_rusentrel.joined.experiment import RuSentRelWithRuAttitudesExperiment
from arekit.contrib.experiment_rusentrel.sl.experiment import RuSentRelExperiment
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes


def create_experiment(exp_type,
                      experiment_data,
                      folding_type,
                      rusentrel_version,
                      load_ruattitude_docs,
                      experiment_io_type,
                      extra_name_suffix,
                      do_log=True,
                      ruattitudes_version=None):
    """ This method allows to instanciate all the supported experiments
        by `contrib/experiments/` module of AREkit framework.
    """

    assert(isinstance(exp_type, ExperimentTypes))
    assert(isinstance(experiment_data, DataIO))
    assert(isinstance(folding_type, FoldingType))
    assert(isinstance(load_ruattitude_docs, bool))

    if exp_type == ExperimentTypes.RuSentRel:
        # Supervised learning experiment type.
        return RuSentRelExperiment(exp_data=experiment_data,
                                   version=rusentrel_version,
                                   folding_type=folding_type,
                                   experiment_io_type=experiment_io_type,
                                   extra_name_suffix=extra_name_suffix,
                                   do_log=do_log)

    if exp_type == ExperimentTypes.RuAttitudes:
        # Application of the distant supervision only (assumes for pretraining purposes)
        return RuAttitudesExperiment(exp_data=experiment_data,
                                     version=ruattitudes_version,
                                     experiment_io_type=experiment_io_type,
                                     load_docs=load_ruattitude_docs,
                                     extra_name_suffix=extra_name_suffix,
                                     do_log=do_log)

    if exp_type == ExperimentTypes.RuSentRelWithRuAttitudes:
        # Supervised learning with an application of distant supervision in training process.
        return RuSentRelWithRuAttitudesExperiment(ruattitudes_version=ruattitudes_version,
                                                  exp_data=experiment_data,
                                                  rusentrel_version=rusentrel_version,
                                                  folding_type=folding_type,
                                                  experiment_io_type=experiment_io_type,
                                                  load_docs=load_ruattitude_docs,
                                                  extra_name_suffix=extra_name_suffix,
                                                  do_log=do_log)
