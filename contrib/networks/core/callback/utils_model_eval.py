import logging

from arekit.common.experiment import const
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.output.multiple import MulticlassOutput
from arekit.common.experiment.output.opinions.converter import OutputToOpinionCollectionsConverter
from arekit.common.experiment.output.opinions.writer import save_opinion_collections
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.opinions.formatter import OpinionCollectionsFormatter
from arekit.common.utils import progress_bar_iter
from arekit.contrib.networks.core.callback.utils_hidden_states import save_minibatch_all_input_dependent_hidden_values
from arekit.contrib.networks.core.data_handling.predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.input.readers.samples import NetworkInputSampleReader
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.output.encoder import NetworkOutputEncoder
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter

logger = logging.getLogger(__name__)


def evaluate_model(experiment, data_type, epoch_index, model,
                   labels_formatter, save_hidden_params, log_dir):
    """ Performs Model Evaluation on a particular state (i.e. epoch),
        for a particular data type.
    """
    assert(isinstance(labels_formatter, RuSentRelLabelsFormatter))
    assert(isinstance(model, BaseTensorflowModel))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    assert(isinstance(save_hidden_params, bool))

    # Prediction result is a pair of the following parameters:
    # idhp -- input dependent variables that might be saved for additional research.
    idhp = model.predict(data_type=data_type)

    assert (isinstance(idhp, NetworkInputDependentVariables))

    # Getting access to the original samples with the related reader.
    # In this scenario, there is a need to obtain news ids for the related sample_id.
    samples_filepath = experiment.ExperimentIO.get_input_sample_filepath(data_type=data_type)
    samples_reader = NetworkInputSampleReader.from_tsv(samples_filepath, MultipleIDProvider())
    news_id_by_sample_id = samples_reader.calculate_news_id_by_sample_id_dict()

    # Create and save output.
    result_filepath = experiment.ExperimentIO.get_output_model_results_filepath(data_type=data_type,
                                                                                epoch_index=epoch_index)
    logger.info(u"Target output filepath: {}".format(result_filepath))
    labeling_collection = model.get_samples_labeling_collection(data_type=data_type)
    sample_id_with_labels_iter = labeling_collection.iter_non_duplicated_labeled_sample_row_ids()
    NetworkOutputEncoder.to_tsv(
        filepath=result_filepath,
        sample_id_with_labels_iter=__log_wrap_samples_iter(sample_id_with_labels_iter),
        column_extra_funcs=[
            (const.NEWS_ID, lambda sample_id: news_id_by_sample_id[sample_id])
        ],
        labels_scaler=experiment.DataIO.LabelsScaler)

    # Convert output to result.
    __convert_output_to_opinion_collections(
        exp_io=experiment.ExperimentIO,
        opin_ops=experiment.OpinionOperations,
        doc_ops=experiment.DocumentOperations,
        labels_scaler=experiment.DataIO.LabelsScaler,
        opin_fmt=experiment.DataIO.OpinionFormatter,
        data_type=data_type,
        epoch_index=epoch_index,
        result_filepath=result_filepath,
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


def __convert_output_to_opinion_collections(exp_io, opin_ops, doc_ops, labels_scaler, opin_fmt,
                                            result_filepath, data_type, epoch_index,
                                            labels_formatter):
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(labels_scaler, BaseLabelScaler))
    assert(isinstance(exp_io, NetworkIOUtils))
    assert(isinstance(opin_fmt, OpinionCollectionsFormatter))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    assert(isinstance(labels_formatter, StringLabelsFormatter))

    opinions_source = exp_io.get_input_opinions_filepath(data_type=data_type)

    cmp_doc_ids_set = set(doc_ops.iter_doc_ids_to_compare())

    # Extract iterator.
    collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        output_filepath=result_filepath,
        opinions_reader=InputOpinionReader.from_tsv(opinions_source),
        labels_scaler=labels_scaler,
        opinion_operations=opin_ops,
        keep_doc_id_func=lambda doc_id: doc_id in cmp_doc_ids_set,
        # TODO. bring this onto parameters level.
        label_calculation_mode=LabelCalculationMode.AVERAGE,
        output=MulticlassOutput(labels_scaler),
        keep_news_ids_from_samples_reader=True,
        keep_ids_from_samples_reader=False)

    # Save collection.
    save_opinion_collections(
        opinion_collection_iter=__log_wrap_collections_conversion_iter(collections_iter),
        create_file_func=lambda doc_id: exp_io.create_result_opinion_collection_filepath(data_type=data_type,
                                                                                         doc_id=doc_id,
                                                                                         epoch_index=epoch_index),
        save_to_file_func=lambda filepath, collection: opin_fmt.save_to_file(collection=collection,
                                                                             filepath=filepath,
                                                                             labels_formatter=labels_formatter))


def __log_wrap_samples_iter(it):
    return progress_bar_iter(iterable=it,
                             desc=u'Writing output',
                             unit=u'rows')


def __log_wrap_collections_conversion_iter(it):
    return progress_bar_iter(iterable=it,
                             desc=u"Converting: Output Rows -> Opinion Collections",
                             unit=u"colls")
