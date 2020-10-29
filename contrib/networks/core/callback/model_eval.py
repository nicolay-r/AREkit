from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.output.multiple import MulticlassOutput
from arekit.common.experiment.output.opinions.converter import OutputToOpinionCollectionsConverter
from arekit.common.experiment.output.opinions.writer import save_opinion_collections
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.contrib.networks.core.input.readers.samples import NetworkInputSampleReader


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
        experiment=experiment,
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
