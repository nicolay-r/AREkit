from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.networks.core.pipeline_epoch import EpochHandlingPipelineItem


class EpochLabelsPredictorPipelineItem(EpochHandlingPipelineItem):

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        minibatch = pipeline_ctx.provide("src")

        feed_dict = self._context.create_feed_dict(minibatch=minibatch, data_type=self._data_type)
        uint_labels = self._context.Session.run(self._context.Network.Labels, feed_dict=feed_dict)

        pipeline_ctx.update("uint_labels", uint_labels)
