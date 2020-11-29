from arekit.common.experiment.data_type import DataType
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
from arekit.contrib.networks.core.callback.utils_hidden_states import save_minibatch_all_input_dependent_hidden_values
from arekit.contrib.networks.core.data_handling.predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.input.readers.samples import NetworkInputSampleReader
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.output.encoder import NetworkOutputEncoder
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter


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
    # output -- output encoder of the network.
    idhp, output = model.predict(data_type=data_type)

    assert (isinstance(idhp, NetworkInputDependentVariables))
    assert (isinstance(output, NetworkOutputEncoder))

    # Create output filepath
    result_filepath = experiment.ExperimentIO.get_output_model_results_filepath(
        data_type=data_type,
        epoch_index=epoch_index)

    # Save output
    output.to_tsv(filepath=result_filepath,
                  labels_scaler=experiment.DataIO.LabelsScaler)

    # Convert output to result.
    __convert_output_to_opinion_collections(
        exp_io=experiment.ExperimentIO,
        opin_ops=experiment.OpinionOperations,
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


def __convert_output_to_opinion_collections(exp_io, opin_ops, labels_scaler, opin_fmt,
                                            result_filepath, data_type, epoch_index,
                                            labels_formatter):
    assert(isinstance(opin_ops, OpinionOperations))
    assert(isinstance(labels_scaler, BaseLabelScaler))
    assert(isinstance(exp_io, NetworkIOUtils))
    assert(isinstance(opin_fmt, OpinionCollectionsFormatter))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    assert(isinstance(labels_formatter, StringLabelsFormatter))

    samples_filepath = exp_io.get_input_sample_filepath(data_type=data_type)
    opinions_source = exp_io.get_input_opinions_filepath(data_type=data_type)

    # Extract iterator.
    collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        output_filepath=result_filepath,
        opinions_reader=InputOpinionReader.from_tsv(opinions_source),
        samples_reader=NetworkInputSampleReader.from_tsv(samples_filepath, MultipleIDProvider()),
        labels_scaler=labels_scaler,
        opinion_operations=opin_ops,
        label_calculation_mode=LabelCalculationMode.AVERAGE,
        output=MulticlassOutput(labels_scaler),
        keep_news_ids_from_samples_reader=True,
        keep_ids_from_samples_reader=False)

    # Save collection.
    save_opinion_collections(
        opinion_collection_iter=collections_iter,
        create_file_func=lambda doc_id: exp_io.create_result_opinion_collection_filepath(data_type=data_type,
                                                                                         doc_id=doc_id,
                                                                                         epoch_index=epoch_index),
        save_to_file_func=lambda filepath, collection: opin_fmt.save_to_file(collection=collection,
                                                                             filepath=filepath,
                                                                             labels_formatter=labels_formatter))
