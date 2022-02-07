import logging

from arekit.contrib.networks.core.callback.base import NetworkCallback
from arekit.contrib.networks.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.core.utils import get_item_from_pipeline

logger = logging.getLogger(__name__)


class ResultWriterCallback(NetworkCallback):

    def __init__(self, result_filepath, labels_scaler):
        assert(isinstance(result_filepath, str))
        self.__result_filepath = result_filepath
        self.__labels_scaler = labels_scaler

    def on_predict_finished(self, pipeline):
        super(ResultWriterCallback, self).on_predict_finished(pipeline)

        item = get_item_from_pipeline(pipeline=pipeline, item_type=EpochLabelsCollectorPipelineItem)
        labeled_samples = item.LabeledSamples
        predict_provider = BasePredictProvider()

        with TsvPredictWriter(filepath=self.__result_filepath) as out:
            title, contents_it = predict_provider.provide(
                sample_id_with_uint_labels_iter=labeled_samples.iter_non_duplicated_labeled_sample_row_ids(),
                labels_scaler=self.__labels_scaler)

            out.write(title=title, contents_it=contents_it)
