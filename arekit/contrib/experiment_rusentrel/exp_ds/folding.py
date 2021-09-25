from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding


def create_ruattitudes_experiment_data_folding(doc_ids_to_fold):
    supported_data_types = [DataType.Train]
    return NoFolding(doc_ids_to_fold=doc_ids_to_fold,
                     supported_data_types=supported_data_types)
