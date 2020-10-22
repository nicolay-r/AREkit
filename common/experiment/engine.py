import logging
from os.path import join, exists
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.io_utils import BaseIOUtils


class BaseExperimentEngine(object):

    @staticmethod
    def _mark_dir_for_serialization(io_utils, logger, experiment, skip_if_folder_exists):
        assert(issubclass(io_utils, BaseIOUtils))
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(skip_if_folder_exists, bool))

        target_dir = io_utils.get_target_dir(experiment)
        target_file = join(target_dir, 'lock.txt')
        if exists(target_file) and skip_if_folder_exists:
            logger.info("TARGET DIR EXISTS: {}".format(target_dir))
            return
        else:
            open(target_file, 'a').close()

    @staticmethod
    def _setup_logger():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(stream_handler)
        return logger

    @staticmethod
    def _perform_neutral_annotation(logger, experiment):
        assert(isinstance(experiment, BaseExperiment))

        # Initializing annotator
        logger.info("Initializing neutral annotator ...")
        experiment.initialize_neutral_annotator()

        # Perform neutral annotation
        logger.info("Perform neutral annotation ...")
        for data_type in experiment.DocumentOperations.iter_supported_data_types():
            experiment.NeutralAnnotator.create_collection(data_type=data_type)

    # TODO. Use this function with handler parameter.
    # TODO. Where handler performs actions per every cv iteration.
    # TODO. (This is called both from experiments and serializers).
    @staticmethod
    def _iter_cv_index(experiment):
        """ Performs an update of cv_folding algorithm state for every cv_iteration.
        """
        for cv_index in range(experiment.DataIO.CVFoldingAlgorithm.CVCount):
            experiment.DataIO.CVFoldingAlgorithm.set_iteration_index(cv_index)
            yield cv_index
