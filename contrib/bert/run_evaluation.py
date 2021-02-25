from os.path import exists, join

from arekit.common.experiment.data.training import TrainingData
from arekit.common.experiment.engine.cv_based import ExperimentEngine
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.input.readers.opinion import InputOpinionReader
from arekit.common.experiment.input.readers.sample import InputSampleReader
from arekit.common.experiment.output.opinions.converter import OutputToOpinionCollectionsConverter
from arekit.common.experiment.output.opinions.writer import save_opinion_collections
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.contrib.bert.callback import Callback
from arekit.contrib.bert.output.eval_helper import EvalHelper
from arekit.contrib.bert.output.google_bert import GoogleBertMulticlassOutput
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter


class LanguageModelExperimentEvaluator(ExperimentEngine):

    def __init__(self, experiment, data_type, eval_helper, max_epochs_count):
        assert(isinstance(eval_helper, EvalHelper))
        assert(isinstance(max_epochs_count, int))

        super(LanguageModelExperimentEvaluator, self).__init__(experiment)

        self.__data_type = data_type
        self.__eval_helper = eval_helper
        self.__max_epochs_count = max_epochs_count

    def __get_target_dir(self):
        return self._experiment.ExperimentIO.get_target_dir()

    def _handle_iteration(self, iter_index):
        exp_io = self._experiment.ExperimentIO
        exp_data = self._experiment.DataIO
        assert(isinstance(exp_data, TrainingData))

        # Setup callback.
        callback = exp_data.Callback
        assert(isinstance(callback, Callback))
        callback.set_iter_index(iter_index)

        # Providing opinions reader.
        opinions_tsv_filepath = exp_io.get_input_opinions_filepath(self.__data_type)
        # Providing samples reader.
        samples_tsv_filepath = exp_io.get_input_sample_filepath(self.__data_type)

        row_id_provider = MultipleIDProvider()
        labels_formatter = RuSentRelLabelsFormatter()
        cmp_doc_ids_set = set(self._experiment.DocumentOperations.iter_doc_ids_to_compare())

        with callback:
            for epoch_index in range(self.__max_epochs_count):

                result_filename_template = self.__eval_helper.get_results_filename(
                    iter_index=iter_index,
                    epoch_index=epoch_index)

                result_filepath = join(self.__get_target_dir(), result_filename_template)

                if not exists(result_filepath):
                    continue

                print "Found:", result_filepath

                # We utilize google bert format, where every row
                # consist of label probabilities per every class
                output = GoogleBertMulticlassOutput(
                    labels_scaler=exp_data.LabelsScaler,
                    samples_reader=InputSampleReader.from_tsv(filepath=samples_tsv_filepath,
                                                              row_ids_provider=row_id_provider),
                    has_output_header=False)

                # iterate opinion collections.
                collections_iter = OutputToOpinionCollectionsConverter.iter_opinion_collections(
                    output_filepath=result_filepath,
                    opinions_reader=InputOpinionReader.from_tsv(opinions_tsv_filepath, compression='infer'),
                    labels_scaler=exp_data.LabelsScaler,
                    create_opinion_collection_func=self._experiment.OpinionOperations.create_opinion_collection,
                    keep_doc_id_func=lambda doc_id: doc_id in cmp_doc_ids_set,
                    label_calculation_mode=LabelCalculationMode.AVERAGE,
                    output=output)

                save_opinion_collections(
                    opinion_collection_iter=collections_iter,
                    create_file_func=lambda doc_id: exp_io.create_result_opinion_collection_filepath(
                        data_type=self.__data_type,
                        doc_id=doc_id,
                        epoch_index=epoch_index),
                    save_to_file_func=lambda filepath, collection:
                        self._experiment.DataIO.OpinionFormatter.save_to_file(
                            collection=collection,
                            filepath=filepath,
                            labels_formatter=labels_formatter))

                # evaluate
                result = self._experiment.evaluate(data_type=self.__data_type,
                                                   epoch_index=epoch_index)
                result.calculate()

                # saving results.
                callback.write_results(result=result,
                                       data_type=self.__data_type,
                                       epoch_index=epoch_index)

    def _before_running(self):
        # Providing a root dir for logging.
        callback = self._experiment.DataIO.Callback
        callback.set_log_dir(self.__get_target_dir())
