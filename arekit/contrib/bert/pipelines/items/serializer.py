from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.utils.utils_folding import folding_iter_states
from arekit.contrib.utils.serializer import InputDataSerializationHelper


class BertExperimentInputSerializerPipelineItem(BasePipelineItem):

    def __init__(self, data_type_pipelines, sample_rows_provider, exp_io,
                 data_folding, save_labels_func, balance_func, keep_opinions_repo=False):
        """ sample_rows_formatter:
                how we format input texts for a BERT model, for example:
                    - single text
                    - two sequences, separated by [SEP] token

            save_labels_func: function
                data_type -> bool

            data_type_pipelines: dict of, for example:
                {
                    DataType.Train: BasePipeline,
                    DataType.Test: BasePipeline
                }

                pipeline: doc_id -> parsed_news -> annot -> opinion linkages
                    for example, function: sentiment_attitude_extraction_default_pipeline
        """
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(data_folding, BaseDataFolding))
        super(BertExperimentInputSerializerPipelineItem, self).__init__()

        self.__sample_rows_provider = sample_rows_provider
        self.__balance_func = balance_func
        self.__exp_io = exp_io
        self.__data_type_pipelines = data_type_pipelines
        self.__save_labels_func = save_labels_func
        self.__keep_opinions_repo = keep_opinions_repo
        self.__data_folding = data_folding

    # region private methods

    def __serialize_iteration(self, data_type, pipeline, data_folding):
        assert(isinstance(data_type, DataType))

        repos = {
            "sample": InputDataSerializationHelper.create_samples_repo(
                keep_labels=self.__save_labels_func(data_type),
                rows_provider=self.__sample_rows_provider),
            "opinion": InputDataSerializationHelper.create_opinion_repo()
        }

        writer_and_targets = {
            "sample": (self.__exp_io.create_samples_writer(),
                       self.__exp_io.create_samples_writer_target(
                           data_type=data_type, data_folding=data_folding)),
            "opinion": (self.__exp_io.create_opinions_writer(),
                        self.__exp_io.create_opinions_writer_target(
                            data_type=data_type, data_folding=data_folding))
        }

        for description, repo in repos.items():

            if description == "opinion" and not self.__keep_opinions_repo:
                continue

            InputDataSerializationHelper.fill_and_write(
                repo=repo,
                pipeline=pipeline,
                doc_ids_iter=data_folding.fold_doc_ids_set()[data_type],
                do_balance=self.__balance_func(data_type),
                desc="{desc} [{data_type}]".format(desc=description, data_type=data_type),
                writer=writer_and_targets[description][0],
                target=writer_and_targets[description][1])

    def __handle_iteration(self, data_type_pipelines, data_folding):
        """ Performing data serialization for a particular iteration
        """
        for data_type, pipeline in data_type_pipelines.items():
            self.__serialize_iteration(data_type=data_type, pipeline=pipeline, data_folding=data_folding)

    # endregion

    def apply_core(self, input_data, pipeline_ctx=None):
        for _ in folding_iter_states(self.__data_folding):
            self.__handle_iteration(data_type_pipelines=self.__data_type_pipelines,
                                    data_folding=self.__data_folding)
