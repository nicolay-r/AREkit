import logging

from arekit.common.data import const
from arekit.common.data.views.linkages.multilabel import MultilableOpinionLinkagesView
from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.pipelines.opinion_collections import output_to_opinion_collections
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item_handle import HandleIterPipelineItem
from arekit.common.utils import progress_bar_iter
from arekit.contrib.networks.core.callback.utils_hidden_states import save_minibatch_all_input_dependent_hidden_values
from arekit.contrib.networks.core.ctx_predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.model import BaseTensorflowModel

from arekit.contrib.networks.core.predict.tsv_provider import TsvPredictProvider
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter

logger = logging.getLogger(__name__)


def evaluate_model(experiment, label_scaler, data_type, epoch_index, model,
                   labels_formatter, save_hidden_params,
                   label_calc_mode, log_dir):
    """ Performs Model Evaluation on a particular state (i.e. epoch),
        for a particular data type.
    """
    assert(isinstance(labels_formatter, RuSentRelLabelsFormatter))
    assert(isinstance(label_scaler, BaseLabelScaler))
    assert(isinstance(model, BaseTensorflowModel))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    assert(isinstance(save_hidden_params, bool))

    # Prediction result is a pair of the following parameters:
    # idhp -- input dependent variables that might be saved for additional research.
    idhp = model.predict(data_type=data_type)

    assert (isinstance(idhp, NetworkInputDependentVariables))

    samples_view = experiment.ExperimentIO.create_samples_view(data_type)

    # TODO. Filepath-dependency should be removed!
    # Create and save output.
    result_filepath = experiment.ExperimentIO.get_output_model_results_filepath(data_type=data_type,
                                                                                epoch_index=epoch_index)
    logger.info("Target output filepath: {}".format(result_filepath))
    labeled_samples = model.get_labeled_samples_collection(data_type=data_type)
    sample_id_with_uint_labels_iter = labeled_samples.iter_non_duplicated_labeled_sample_row_ids()

    # TODO. This is a limitation, as we focus only tsv.
    doc_id_by_sample_id = __calculate_doc_id_by_sample_id_dict(samples_view.iter_rows(None))
    with TsvPredictProvider(filepath=result_filepath) as out:
        out.load(sample_id_with_uint_labels_iter=__log_wrap_samples_iter(sample_id_with_uint_labels_iter),
                 column_extra_funcs=[(const.DOC_ID, lambda sample_id: doc_id_by_sample_id[sample_id])],
                 labels_scaler=label_scaler)

    # TODO. Pass here the original storage. (NO API for now out there).
    storage = None

    output_view = MultilableOpinionLinkagesView(
        labels_scaler=label_scaler,
        storage=storage)

    # Convert output to result.
    ppl = output_to_opinion_collections(
        iter_opinion_linkages_func=lambda doc_id: output_view.iter_opinion_linkages(
            doc_id=doc_id,
            opinions_view=experiment.ExperimentIO.create_opinions_view(data_type)),
        doc_ids_set=set(experiment.DocumentOperations.iter_tagget_doc_ids(BaseDocumentTag.Compare)),
        opin_ops=experiment.OpinionOperations,
        labels_scaler=label_scaler,
        label_calc_mode=label_calc_mode,
        supported_labels=None)

    # Writing opinion collection.
    save_item = HandleIterPipelineItem(
        lambda doc_id, collection:
        experiment.ExperimentIO.write_opinion_collection(
            collection=collection,
            labels_formatter=labels_formatter,
            target=experiment.ExperimentIO.create_result_opinion_collection_target(
                data_type=data_type,
                epoch_index=epoch_index,
                doc_id=doc_id)))

    # Executing pipeline.
    ppl.append(save_item)
    pipeline_ctx = PipelineContext({
        "src": set(storage.iter_column_values(column_name=const.DOC_ID))
    })
    ppl.run(pipeline_ctx)

    # iterate over the result.
    for _ in pipeline_ctx.provide("src"):
        pass

    # Evaluate.
    result = experiment.evaluate(data_type=data_type,
                                 epoch_index=epoch_index)

    # optionally save input-dependent hidden parameters.
    if save_hidden_params:
        save_minibatch_all_input_dependent_hidden_values(
            predict_log=idhp,
            data_type=data_type,
            log_dir=log_dir,
            epoch_index=epoch_index)

    return result


def __calculate_doc_id_by_sample_id_dict(rows_iter):
    """
    Iter sample_ids with the related labels (if the latter presented in dataframe)
    """
    d = {}

    for row_index, row in rows_iter:

        sample_id = row[const.ID]

        if sample_id in d:
            continue

        d[sample_id] = row[const.DOC_ID]

    return d


def __log_wrap_samples_iter(it):
    return progress_bar_iter(iterable=it,
                             desc='Writing output',
                             unit='rows')


def __log_wrap_collections_conversion_iter(it):
    return progress_bar_iter(iterable=it,
                             desc="Converting: Output Rows -> Opinion Collections",
                             unit="colls")
