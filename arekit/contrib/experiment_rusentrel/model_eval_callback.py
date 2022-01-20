import logging

from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.linkages.multilabel import MultilableOpinionLinkagesView
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.pipelines.opinion_collections import output_to_opinion_collections_pipeline
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item_handle import HandleIterPipelineItem

from arekit.contrib.networks.core.idhv_collection import NetworkInputDependentVariables
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.core.predict.provider import BasePredictProvider
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter

logger = logging.getLogger(__name__)


# TODO. split onto callback items.
def evaluate_model(experiment, label_scaler, data_type, epoch_index, model,
                   labels_formatter, label_calc_mode):
    """ Performs Model Evaluation on a particular state (i.e. epoch),
        for a particular data type.
    """
    assert(isinstance(labels_formatter, RuSentRelLabelsFormatter))
    assert(isinstance(label_scaler, BaseLabelScaler))
    assert(isinstance(model, BaseTensorflowModel))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))

    # Prediction result is a pair of the following parameters:
    # idhp -- input dependent variables that might be saved for additional research.
    idhp = model.predict(data_type=data_type)

    assert (isinstance(idhp, NetworkInputDependentVariables))

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

    storage = BaseRowsStorage.from_tsv(filepath=result_filepath)
    ######################################################################################################

    linkages_view = MultilableOpinionLinkagesView(
        labels_scaler=label_scaler,
        storage=storage)

    # Convert output to result.
    ppl = output_to_opinion_collections_pipeline(
        iter_opinion_linkages_func=lambda doc_id: linkages_view.iter_opinion_linkages(
            doc_id=doc_id,
            opinions_view=experiment.ExperimentIO.create_opinions_view(data_type)),
        doc_ids_set=set(experiment.DocumentOperations.iter_tagget_doc_ids(BaseDocumentTag.Compare)),
        create_opinion_collection_func=experiment.OpinionOperations.create_opinion_collection,
        labels_scaler=label_scaler,
        label_calc_mode=label_calc_mode)

    # Writing opinion collection.
    save_item = HandleIterPipelineItem(
        lambda data:
        experiment.ExperimentIO.write_opinion_collection(
            collection=data[1],
            labels_formatter=labels_formatter,
            target=experiment.ExperimentIO.create_result_opinion_collection_target(
                data_type=data_type,
                epoch_index=epoch_index,
                doc_id=data[0])))

    # Executing pipeline.
    ppl.append(save_item)
    pipeline_ctx = PipelineContext({
        "src": set(storage.iter_column_values(column_name=const.DOC_ID))
    })
    ppl.run(pipeline_ctx)

    # iterate over the result.
    for _ in pipeline_ctx.provide("src"):
        pass

    # TODO. Callback evaluator.
    # TODO. This is an experiment callback.
    result = experiment.evaluate(data_type=data_type,
                                 epoch_index=epoch_index)


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
