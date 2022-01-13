from arekit.contrib.experiment_rusentrel.model_io.tf_networks import RuSentRelExperimentNetworkIOUtils
from examples.network.args.const import DATA_DIR


class CustomRuSentRelNetworkExperimentIO(RuSentRelExperimentNetworkIOUtils):

    def try_prepare(self):
        pass

    def _get_experiment_sources_dir(self):
        return DATA_DIR
