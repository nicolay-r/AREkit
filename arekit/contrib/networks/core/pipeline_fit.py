import numpy as np

from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.networks.core.pipeline_epoch import EpochHandlingPipelineItem


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

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        minibatch = pipeline_ctx.provide("src")

        feed_dict = self._context.create_feed_dict(minibatch=minibatch, data_type=self._data_type)

        hidden_list = list(self._context.Network.iter_hidden_parameters())
        fetches_default = [self._context.Optimiser, self._context.Network.Cost, self._context.Network.Accuracy]
        fetches_hidden = [tensor for _, tensor in hidden_list]

        result = self._context.Session.run(fetches_default + fetches_hidden, feed_dict=feed_dict)

        cost = result[1]

        self.__fit_total_cost += np.mean(cost)
        self.__fit_total_acc += result[2]
        self.__groups_count += 1
