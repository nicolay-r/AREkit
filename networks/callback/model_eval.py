from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.encoder import BaseInputEncoder
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.output.multiple import MulticlassOutput
from arekit.common.experiment.output.opinions.converter import OutputToOpinionCollectionsConverter
from arekit.common.experiment.output.opinions.writer import save_collections
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode


def perform_evaluation(tsv_results_filepath, source_dir, template,
                       data_type, experiment, epoch_index, labels_formatter):
    """
    1. Converting results
    2. Perform evaluation.
    """
    assert(isinstance(tsv_results_filepath, unicode))
    assert(isinstance(template, unicode))
    assert(isinstance(data_type, DataType))
    assert(isinstance(experiment, BaseExperiment))
    assert(isinstance(epoch_index, int))
    assert(isinstance(labels_formatter, StringLabelsFormatter))

    # Reading opinions.
    opinions_source, _ = BaseInputEncoder.get_filepaths(out_dir=experiment.get_input_samples_dir(),
                                                        experiment=experiment,
                                                        data_type=data_type)

    # Extract iterator.
    collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
        source_dir=source_dir,
        filename_template=template,
        opinions_reader=InputOpinionReader.from_tsv(opinions_source),
        samples_reader=None,
        experiment=experiment,
        label_calculation_mode=LabelCalculationMode.AVERAGE,
        output=MulticlassOutput(experiment.DataIO.LabelsScaler))

    # Save collection.
    save_collections(opinion_collection_iter=collections_iter,
                     experiment=experiment,
                     data_type=data_type,
                     labels_formatter=labels_formatter)

    # Evaluate.
    return experiment.evaluate(data_type=data_type, epoch_index=epoch_index)