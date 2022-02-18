import numpy as np

from arekit.contrib.networks.core.feeding.batch.base import MiniBatch
from arekit.contrib.networks.core.pipeline.item_base import EpochHandlingPipelineItem


class MinibatchFittingPipelineItem(EpochHandlingPipelineItem):
    
    def __init__(self):
        super(MinibatchFittingPipelineItem, self).__init__()
        self.__fit_total_cost = None
        self.__fit_total_acc = None
        self.__groups_count = None

    @property
    def TotalFitCost(self):
        return self.__fit_total_cost / self.__groups_count

    @property
    def TotalFitAccuracy(self):
        return self.__fit_total_acc / self.__groups_count

    def before_epoch(self, model_context, data_type):
        super(MinibatchFittingPipelineItem, self).before_epoch(model_context=model_context,
                                                               data_type=data_type)
        self.__fit_total_cost = 0
        self.__fit_total_acc = 0
        self.__groups_count = 0

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, MiniBatch))

        feed_dict = self._context.create_feed_dict(minibatch=input_data, data_type=self._data_type)

        hidden_list = list(self._context.Network.iter_hidden_parameters())
        fetches_default = [self._context.Optimiser, self._context.Network.Cost, self._context.Network.Accuracy]
        fetches_hidden = [tensor for _, tensor in hidden_list]

        result = self._context.Session.run(fetches_default + fetches_hidden, feed_dict=feed_dict)

        cost = result[1]

        self.__fit_total_cost += np.mean(cost)
        self.__fit_total_acc += result[2]
        self.__groups_count += 1
