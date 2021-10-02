import logging

from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.output_multiple import MulticlassOutputView
from arekit.common.experiment import const
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.provider import OpinionCollectionsProvider
from arekit.common.utils import progress_bar_iter
from arekit.contrib.networks.core.callback.utils_hidden_states import save_minibatch_all_input_dependent_hidden_values
from arekit.contrib.networks.core.ctx_predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
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
    news_id_by_sample_id = samples_view.calculate_news_id_by_sample_id_dict()

    # TODO. Filepath-dependency should be removed!
    # Create and save output.
    result_filepath = experiment.ExperimentIO.get_output_model_results_filepath(data_type=data_type,
                                                                                epoch_index=epoch_index)
    logger.info("Target output filepath: {}".format(result_filepath))
    labeled_samples = model.get_labeled_samples_collection(data_type=data_type)
    sample_id_with_uint_labels_iter = labeled_samples.iter_non_duplicated_labeled_sample_row_ids()

    # TODO. This is a limitation, as we focus only tsv.
    with TsvPredictProvider(filepath=result_filepath) as out:
        out.load(sample_id_with_uint_labels_iter=__log_wrap_samples_iter(sample_id_with_uint_labels_iter),
                 column_extra_funcs=[(const.NEWS_ID, lambda sample_id: news_id_by_sample_id[sample_id])],
                 labels_scaler=label_scaler)

    # Convert output to result.
    __convert_output_to_opinion_collections(
        exp_io=experiment.ExperimentIO,
        opin_ops=experiment.OpinionOperations,
        doc_ops=experiment.DocumentOperations,
        labels_scaler=label_scaler,
        opin_provider=experiment.ExperimentIO.OpinionCollectionProvider,
        supported_collection_labels=experiment.DataIO.SupportedCollectionLabels,
        data_type=data_type,
        epoch_index=epoch_index,
        output_storage=BaseRowsStorage.from_tsv(result_filepath),
        label_calc_mode=label_calc_mode,
        labels_formatter=labels_formatter)

    # Evaluate.
    result = experiment.evaluate(data_type=data_type,
                                 epoch_index=epoch_index)

    # optionally save input-dependend hidden parameters.
    if save_hidden_params:
        save_minibatch_all_input_dependent_hidden_values(
            predict_log=idhp,
            data_type=data_type,
            log_dir=log_dir,
            epoch_index=epoch_index)

    return result


# TODO. Pass TsvInputOpinionReader.
def __convert_output_to_opinion_collections(exp_io, opin_ops, doc_ops, labels_scaler, opin_provider,
                                            output_storage, data_type, epoch_index,
                                            supported_collection_labels, label_calc_mode, labels_formatter):
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(labels_scaler, BaseLabelScaler))
    assert(isinstance(exp_io, NetworkIOUtils))
    assert(isinstance(opin_provider, OpinionCollectionsProvider))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    assert(isinstance(label_calc_mode, LabelCalculationMode))
    assert(isinstance(labels_formatter, StringLabelsFormatter))

    cmp_doc_ids_set = set(doc_ops.iter_doc_ids_to_compare())

    output_view = MulticlassOutputView(labels_scaler=labels_scaler,
                                       storage=output_storage)

    # Extract iterator.
    collections_iter = output_view.iter_opinion_collections(
        opinions_view=exp_io.create_opinions_view(data_type),
        keep_doc_id_func=lambda doc_id: doc_id in cmp_doc_ids_set,
        to_collection_func=lambda linked_iter: __create_opinion_collection(
            linked_iter=linked_iter,
            supported_labels=supported_collection_labels,
            create_opinion_collection=opin_ops.create_and_fill_opinion_collection,
            label_scaler=labels_scaler))

    # Save collection.
    for doc_id, collection in __log_wrap_collections_conversion_iter(collections_iter):

        target = exp_io.create_result_opinion_collection_target(
            data_type=data_type,
            epoch_index=epoch_index,
            doc_id=doc_id)

        exp_io.serialize_opinion_collection(
            collection=collection,
            doc_id=doc_id,
            data_type=data_type,
            labels_formatter=labels_formatter,
            target=target)


def __create_opinion_collection(linked_iter, supported_labels, label_scaler, create_opinion_collection):
    return create_opinion_collection(
        create_opinion_collection=create_opinion_collection,
        linked_data_iter=linked_iter,
        labels_helper=SingleLabelsHelper(label_scaler),
        to_opinion_func=__create_labeled_opinion,
        label_calc_mode=LabelCalculationMode.AVERAGE,
        supported_labels=supported_labels)


def __create_labeled_opinion(item, label):
    assert(isinstance(item, Opinion))
    return Opinion(source_value=item.SourceValue,
                   target_value=item.TargetValue,
                   sentiment=label)


def __log_wrap_samples_iter(it):
    return progress_bar_iter(iterable=it,
                             desc='Writing output',
                             unit='rows')


def __log_wrap_collections_conversion_iter(it):
    return progress_bar_iter(iterable=it,
                             desc="Converting: Output Rows -> Opinion Collections",
                             unit="colls")
