from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.networks.core.pipeline_epoch import EpochHandlingPipelineItem


class EpochLabelsPredictorPipelineItem(EpochHandlingPipelineItem):
    """ Considering to treat feed dictionary.
    """
    
    def __init__(self):
        super(EpochLabelsPredictorPipelineItem, self).__init__()
        self.__labeled_samples = None

    @property
    def LabeledSamples(self):
        return self.__labeled_samples

    def before_epoch(self, model_context, data_type):
        super(EpochLabelsPredictorPipelineItem, self).before_epoch(model_context=model_context,
                                                                   data_type=data_type)

        # Select the appropriate labels collection.
        self.__labeled_samples = self._context.InferenceContext.LabeledSamplesCollections[data_type]

        # Clear and assert the correctness.
        # TODO. #257. Remove. Create new instance instead.
        self.__labeled_samples.reset_labels()
        # TODO. #257. Remove. Create new instance instead.
        assert(self.__labeled_samples.is_empty())

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        minibatch = pipeline_ctx.provide("src")

        feed_dict = self._context.create_feed_dict(minibatch=minibatch, data_type=self._data_type)
        uint_labels = self._context.Session.run(self._context.Network.Labels, feed_dict=feed_dict)

        # Apply labeling.
        for bag_index, bag in enumerate(minibatch.iter_by_bags()):
            uint_label = int(uint_labels[bag_index])
            for sample in bag:
                self.__labeled_samples.assign_uint_label(uint_label, sample.ID)
