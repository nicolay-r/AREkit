from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.exp_ds.experiment import RuAttitudesExperiment
from arekit.contrib.experiment_rusentrel.exp_joined.experiment import RuSentRelWithRuAttitudesExperiment
from arekit.contrib.experiment_rusentrel.exp_sl.experiment import RuSentRelExperiment
from arekit.contrib.experiment_rusentrel.exp_sl.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes


def create_experiment(exp_type,
                      exp_ctx,
                      exp_io,
                      folding_type,
                      rusentrel_version,
                      load_ruattitude_docs,
                      do_log=True,
                      ruattitudes_version=None):
    """ This method allows to instanciate all the supported experiments
        by `contrib/experiments/` module of AREkit framework.
    """

    assert(isinstance(exp_io, BaseIOUtils))
    assert(isinstance(exp_type, ExperimentTypes))
    assert(isinstance(exp_ctx, ExperimentContext))
    assert(isinstance(folding_type, FoldingType))
    assert(isinstance(load_ruattitude_docs, bool))

    if exp_type == ExperimentTypes.RuSentRel:
        # Supervised learning experiment type.
        return RuSentRelExperiment(exp_ctx=exp_ctx,
                                   version=rusentrel_version,
                                   folding_type=folding_type,
                                   exp_io=exp_io,
                                   do_log=do_log)

    if exp_type == ExperimentTypes.RuAttitudes:
        # Application of the distant supervision only (assumes for pretraining purposes)
        return RuAttitudesExperiment(exp_ctx=exp_ctx,
                                     version=ruattitudes_version,
                                     exp_io=exp_io,
                                     load_docs=load_ruattitude_docs,
                                     do_log=do_log)

    if exp_type == ExperimentTypes.RuSentRelWithRuAttitudes:
        # Supervised learning with an application of distant supervision in training process.
        return RuSentRelWithRuAttitudesExperiment(ruattitudes_version=ruattitudes_version,
                                                  exp_ctx=exp_ctx,
                                                  rusentrel_version=rusentrel_version,
                                                  folding_type=folding_type,
                                                  exp_io=exp_io,
                                                  load_docs=load_ruattitude_docs,
                                                  do_log=do_log)


def create_folding(exp_type, folding_type, rusentrel_version):
    assert(isinstance(exp_type, ExperimentTypes))

    if exp_type == ExperimentTypes.RuSentRel:
        return create_rusentrel_experiment_data_folding(folding_type=folding_type,
                                                        version=rusentrel_version)

    if exp_type == ExperimentTypes.RuAttitudes:
        # ru_attitudes = read_ruattitudes_in_memory(version=version,
        #                                           used_doc_ids_set=None,
        #                                           keep_doc_ids_only=not load_docs)
        # doc_ids_to_fold = ru_attitudes.keys()

        return NoFolding(doc_ids_to_fold=None,
                         supported_data_types=[DataType.Train])

    if exp_type == ExperimentTypes.RuSentRelWithRuAttitudes:
        return create_rusentrel_experiment_data_folding(folding_type=folding_type,
                                                        version=rusentrel_version)
