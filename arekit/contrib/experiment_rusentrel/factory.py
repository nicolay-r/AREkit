from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.folding.types import FoldingType
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.experiment_rusentrel.common import create_text_parser
from arekit.contrib.experiment_rusentrel.exp_ds.factory import create_ruattitudes_experiment
from arekit.contrib.experiment_rusentrel.exp_ds.folding import create_ruattitudes_experiment_data_folding
from arekit.contrib.experiment_rusentrel.exp_sl.factory import create_rusentrel_experiment
from arekit.contrib.experiment_rusentrel.exp_sl.folding import create_rusentrel_experiment_data_folding
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions


def create_experiment(exp_type,
                      exp_ctx,
                      exp_io,
                      folding_type,
                      rusentrel_version,
                      load_ruattitude_docs,
                      ra_doc_id_func,
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
        return create_rusentrel_experiment(exp_ctx=exp_ctx,
                                           version=rusentrel_version,
                                           folding_type=folding_type,
                                           exp_io=exp_io)

    if exp_type == ExperimentTypes.RuAttitudes:
        # Application of the distant supervision only (assumes for pretraining purposes)
        return create_ruattitudes_experiment(exp_ctx=exp_ctx,
                                             version=ruattitudes_version,
                                             exp_io=exp_io,
                                             load_docs=load_ruattitude_docs,
                                             ra_doc_ids_func=ra_doc_id_func)


def create_folding(exp_type, rusentrel_folding_type, rusentrel_version, ruattitudes_version, ra_doc_id_func):
    assert(isinstance(rusentrel_folding_type, FoldingType))
    assert(isinstance(exp_type, ExperimentTypes))
    assert(isinstance(rusentrel_version, RuSentRelVersions))
    assert(isinstance(ruattitudes_version, RuAttitudesVersions))
    assert(callable(ra_doc_id_func))

    if exp_type == ExperimentTypes.RuSentRel:
        return create_rusentrel_experiment_data_folding(folding_type=rusentrel_folding_type,
                                                        version=rusentrel_version)

    if exp_type == ExperimentTypes.RuAttitudes:
        return create_ruattitudes_experiment_data_folding(version=ruattitudes_version,
                                                          doc_id_func=ra_doc_id_func)


def factory_create_text_parser(exp_type, exp_ctx, synonyms, ppl_items):
    assert(isinstance(ppl_items, list))
    assert(isinstance(synonyms, SynonymsCollection))

    if exp_type == ExperimentTypes.RuAttitudes:
        return create_text_parser(exp_ctx=exp_ctx,
                                  entities_parser=RuAttitudesTextEntitiesParser(),
                                  value_to_group_id_func=None,
                                  ppl_items=ppl_items)
    elif exp_type == ExperimentTypes.RuSentRel:
        return create_text_parser(exp_ctx=exp_ctx,
                                  entities_parser=BratTextEntitiesParser(),
                                  value_to_group_id_func=synonyms.get_synonym_group_index,
                                  ppl_items=ppl_items)

    raise Exception('Not supported type "{}"'.format(exp_type))
