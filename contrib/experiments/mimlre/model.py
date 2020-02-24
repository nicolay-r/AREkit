from arekit.networks.multi.training.batch import MultiInstanceBatch
from arekit.contrib.experiments import ContextLevelTensorflowModel
from arekit.contrib.experiments.mimlre.helper.initialization import MIMLREModelInitHelper


class MIMLRETensorflowModel(ContextLevelTensorflowModel):

    def create_batch_by_bags_group(self, bags_group):
        return MultiInstanceBatch(bags_group)

    def create_model_init_helper(self):
        return MIMLREModelInitHelper(io=self.IO, config=self.Config)
