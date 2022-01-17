from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.networks.core.pipeline_epoch import EpochHandlingPipelineItem


class EpochLabelsCollectorPipelineItem(EpochHandlingPipelineItem):

    def __init__(self):
        super(EpochLabelsCollectorPipelineItem, self).__init__()
        self.__labeled_samples = None

    @property
    def LabeledSamples(self):
        return self.__labeled_samples

    def before_epoch(self, model_context, data_type):
        super(EpochLabelsCollectorPipelineItem, self).before_epoch(model_context=model_context,
                                                                   data_type=data_type)
        pairs = self._context.InferenceContext.SampleIdAndLabelPairs[data_type]
        self.__labeled_samples = LabeledCollection(uint_labeled_ids=pairs)

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("uint_labels" in pipeline_ctx)

        minibatch = pipeline_ctx.provide("src")
        uint_labels = pipeline_ctx.provide("uint_labels")

        # Apply labeling.
        for bag_index, bag in enumerate(minibatch.iter_by_bags()):
            uint_label = int(uint_labels[bag_index])
            for sample in bag:
                self.__labeled_samples.assign_uint_label(uint_label, sample.ID)
