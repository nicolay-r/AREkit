from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.contrib.experiment_rusentrel.exp_ds.utils import read_ruattitudes_in_memory
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions


def create_ruattitudes_experiment_data_folding(version, doc_id_func):
    assert(isinstance(version, RuAttitudesVersions))
    ru_attitudes = read_ruattitudes_in_memory(version=version,
                                              doc_id_func=doc_id_func,
                                              keep_doc_ids_only=True)
    return NoFolding(doc_ids_to_fold=list(ru_attitudes.keys()),
                     supported_data_types=[DataType.Train])
