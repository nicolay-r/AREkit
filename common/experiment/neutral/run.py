from arekit.common.experiment.formats.base import BaseExperiment


def perform_neutral_annotation(logger, experiment):
    """ Performing annotation both using annotator and algorithm.
    """
    assert (isinstance(experiment, BaseExperiment))

    # Initializing annotator
    logger.info("Initializing neutral annotator ...")
    experiment.initialize_neutral_annotator()

    # Perform neutral annotation
    logger.info("Perform neutral annotation ...")
    for data_type in experiment.DocumentOperations.iter_supported_data_types():
        experiment.DataIO.NeutralAnnotator.create_collection(data_type=data_type,
                                                             opinion_formatter=experiment.DataIO.OpinionFormatter)
