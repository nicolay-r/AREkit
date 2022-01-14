import os

from arekit.contrib.experiment_rusentrel.model_io.tf_networks import RuSentRelExperimentNetworkIOUtils
from examples.network.args.const import DATA_DIR


class InferIOUtils(RuSentRelExperimentNetworkIOUtils):

    def __create_target(self, doc_id, data_type):
        filename = "result_d{doc_id}_{data_type}.txt".format(doc_id=doc_id, data_type=data_type.name)
        return os.path.join(self._get_target_dir(), filename)

    def _get_experiment_sources_dir(self):
        return DATA_DIR

    def create_opinion_collection_target(self, doc_id, data_type, check_existance=False):
        return self.__create_target(doc_id=doc_id, data_type=data_type)

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        return self.__create_target(doc_id=doc_id, data_type=data_type)
