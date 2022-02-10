import logging

from arekit.contrib.networks.core.callback.base import NetworkCallback
from arekit.contrib.networks.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.core.predict.base_writer import BasePredictWriter
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from arekit.contrib.networks.core.utils import get_item_from_pipeline

logger = logging.getLogger(__name__)


class PredictResultWriterCallback(NetworkCallback):

    def __init__(self, labels_scaler, writer):
        assert(isinstance(writer, BasePredictWriter))
        self.__labels_scaler = labels_scaler
        self.__writer = writer

    def on_predict_finished(self, pipeline):
        super(PredictResultWriterCallback, self).on_predict_finished(pipeline)

        item = get_item_from_pipeline(pipeline=pipeline, item_type=EpochLabelsCollectorPipelineItem)
        labeled_samples = item.LabeledSamples
        predict_provider = BasePredictProvider()

        with self.__writer:
            title, contents_it = predict_provider.provide(
                sample_id_with_uint_labels_iter=labeled_samples.iter_non_duplicated_labeled_sample_row_ids(),
                labels_scaler=self.__labels_scaler)

            self.__writer.write(title=title, contents_it=contents_it)
