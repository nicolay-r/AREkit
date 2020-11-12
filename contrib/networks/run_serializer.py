from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.engine.utils import mark_dir_for_serialization
from arekit.common.experiment.neutral.run import perform_neutral_annotation
from arekit.contrib.networks.core.data_handling.data import HandledData


class NetworksExperimentInputSerializer(ExperimentEngine):

    def __init__(self, experiment, skip_folder_if_exists):

        super(NetworksExperimentInputSerializer, self).__init__(experiment)

        self.__skip_folder_if_exists = skip_folder_if_exists

    # region protected methods

    def _handle_iteration(self, it_index):
        assert(isinstance(self._experiment.DataIO, SerializationData))

        # Performing data serialization.
        if not HandledData.need_serialize(self._experiment):
            return

        # Perform data serialization.
        HandledData.serialize_from_experiment(experiment=self._experiment,
                                              terms_per_context=self._experiment.DataIO.TermsPerContext)

    def _before_running(self):
        perform_neutral_annotation(neutral_annotator=self._experiment.DataIO.NeutralAnnotator,
                                   opin_ops=self._experiment.OpinionOperations,
                                   doc_ops=self._experiment.DocumentOperations,
                                   logger=self._logger)

    # endregion
