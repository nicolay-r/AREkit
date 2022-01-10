import os

from arekit.contrib.experiment_rusentrel.model_io.tf_networks import NetworkIOUtils
from examples.network.args.default import DATA_DIR


class CustomIOUtils(NetworkIOUtils):

    def __create_target(self, doc_id, data_type, epoch_index):
        filename = "result_d{doc_id}_{data_type}_e{epoch_index}.txt".format(
            doc_id=doc_id,
            data_type=data_type.name,
            epoch_index=epoch_index)

        return os.path.join(DATA_DIR, filename)

    def _get_experiment_sources_dir(self):
        return DATA_DIR

    def create_opinion_collection_target(self, doc_id, data_type, check_existance=False):
        return self.__create_target(doc_id=doc_id, data_type=data_type, epoch_index=0)

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        return self.__create_target(doc_id=doc_id, data_type=data_type, epoch_index=epoch_index)