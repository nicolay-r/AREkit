from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.output.multiple import MulticlassOutput
from arekit.common.experiment.output.opinions.converter import OutputToOpinionCollectionsConverter
from arekit.common.experiment.output.opinions.writer import save_opinion_collections
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.contrib.networks.core.callback.utils_hidden_states import save_minibatch_all_input_dependent_hidden_values
from arekit.contrib.networks.core.data_handling.predict_log import NetworkInputDependentVariables
from arekit.contrib.networks.core.input.readers.samples import NetworkInputSampleReader
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

    # Crate filepath
    result_filepath = experiment.ExperimentIO.get_output_model_results_filepath(
        data_type=data_type,
        epoch_index=epoch_index)

    # Save output
    output.to_tsv(filepath=result_filepath)

    # Convert output to result.
    result = perform_experiment_evaluation(
        experiment=experiment,
        data_type=data_type,
        epoch_index=epoch_index,
        labels_formatter=labels_formatter)

    # optionally save input-dependend hidden parameters.
    if save_hidden_params:
        save_minibatch_all_input_dependent_hidden_values(
            predict_log=idhp,
            data_type=data_type,
            log_dir=log_dir,
            epoch_index=epoch_index)

    return result


def perform_experiment_evaluation(experiment, data_type, epoch_index, labels_formatter):
    """
    1. Converting results
    2. Perform evaluation.
    """
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(data_type, DataType))
    assert(isinstance(epoch_index, int))
    assert(isinstance(labels_formatter, StringLabelsFormatter))

    samples_filepath = experiment.ExperimentIO.get_input_sample_filepath(data_type=data_type)
    opinions_source = experiment.ExperimentIO.get_input_opinions_filepath(data_type=data_type)
    result_filepath = experiment.ExperimentIO.get_output_model_results_filepath(data_type=data_type,
                                                                                epoch_index=epoch_index)

    # Extract iterator.
    collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        output_filepath=result_filepath,
        opinions_reader=InputOpinionReader.from_tsv(opinions_source),
        samples_reader=NetworkInputSampleReader.from_tsv(samples_filepath, MultipleIDProvider()),
        labels_scaler=experiment.DataIO.LabelsScaler,
        opinion_operations=experiment.OpinionOperations,
        label_calculation_mode=LabelCalculationMode.AVERAGE,
        output=MulticlassOutput(experiment.DataIO.LabelsScaler),
        keep_news_ids_from_samples_reader=True,
        keep_ids_from_samples_reader=False)

    # Save collection.
    save_opinion_collections(opinion_collection_iter=collections_iter,
                             experiment=experiment,
                             data_type=data_type,
                             labels_formatter=labels_formatter,
                             epoch_index=epoch_index)

    # Evaluate.
    return experiment.evaluate(data_type=data_type, epoch_index=epoch_index)
