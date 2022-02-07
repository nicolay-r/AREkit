import logging

from arekit.common.data import const
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler

from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter

logger = logging.getLogger(__name__)


# TODO. split onto callback items.
def evaluate_model(experiment, label_scaler, data_type, epoch_index, model, labels_formatter):
    """ Performs Model Evaluation on a particular state (i.e. epoch),
        for a particular data type.
    """
    assert(isinstance(labels_formatter, RuSentRelLabelsFormatter))
    assert(isinstance(label_scaler, BaseLabelScaler))
    assert(isinstance(model, BaseTensorflowModel))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))

    model.predict(data_type=data_type)

    samples_view = experiment.ExperimentIO.create_samples_view(data_type)

    # Create and save output.
    ppl_item = model.from_predicted(EpochLabelsCollectorPipelineItem)
    labeled_samples = ppl_item.LabeledSamples
    sample_id_with_uint_labels_iter = labeled_samples.iter_non_duplicated_labeled_sample_row_ids()

    ######################################################################################################
    # TODO. Filepath-dependency should be removed!
    # TODO #168. refactor. Provide storage.
    result_filepath = experiment.ExperimentIO.get_output_model_results_filepath(data_type=data_type,
                                                                                epoch_index=epoch_index)
    logger.info("Target output filepath: {}".format(result_filepath))
    doc_id_by_sample_id = __calculate_doc_id_by_sample_id_dict(samples_view.iter_rows(None))
    predict = BasePredictProvider()
    with TsvPredictWriter(filepath=result_filepath) as out:
        title, contents_it = predict.provide(
            sample_id_with_uint_labels_iter=sample_id_with_uint_labels_iter,
            column_extra_funcs=[(const.DOC_ID, lambda sample_id: doc_id_by_sample_id[sample_id])],
            labels_scaler=label_scaler)
        out.write(title=title, contents_it=contents_it)
    ######################################################################################################


def __calculate_doc_id_by_sample_id_dict(rows_iter):
    """ Iter sample_ids with the related labels (if the latter presented in dataframe)
    """
    d = {}

    for row_index, row in rows_iter:

        sample_id = row[const.ID]

        if sample_id in d:
            continue

        d[sample_id] = row[const.DOC_ID]

    return d
