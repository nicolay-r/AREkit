from arekit.common.experiment.labeling import LabeledCollection
from arekit.contrib.networks.core.feeding.batch.base import MiniBatch
from arekit.contrib.networks.core.pipeline.item_base import EpochHandlingPipelineItem
from arekit.contrib.networks.core.pipeline.item_predict import EpochLabelsPredictorPipelineItem


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
        pairs = self._context.get_sample_id_label_pairs(data_type)
        self.__labeled_samples = LabeledCollection(uint_labeled_ids=pairs)

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, MiniBatch))
        assert(EpochLabelsPredictorPipelineItem.KEY in pipeline_ctx)
        uint_labels = pipeline_ctx.provide(EpochLabelsPredictorPipelineItem.KEY)

        # Apply labeling.
        for bag_index, bag in enumerate(input_data.iter_by_bags()):
            uint_label = int(uint_labels[bag_index])
            for sample in bag:
                self.__labeled_samples.assign_uint_label(uint_label, sample.ID)
