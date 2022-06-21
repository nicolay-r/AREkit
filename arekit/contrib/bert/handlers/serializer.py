from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.contrib.utils.serializer import InputDataSerializationHelper


class BertExperimentInputSerializerIterationHandler(ExperimentIterationHandler):

    def __init__(self, pipeline, sample_rows_provider, exp_io, exp_ctx, doc_ops,
                 save_labels_func, balance_train_samples, data_types):
        """ sample_rows_formatter:
                how we format input texts for a BERT model, for example:
                    - single text
                    - two sequences, separated by [SEP] token
            pipeline:
                doc_id -> parsed_news -> annot -> opinion linkages
                for example, function: sentiment_attitude_extraction_default_pipeline

            save_labels_func: function
                data_type -> bool

            data_types: list
                data_types, for which the data will be generated; required for:
                    - document ids
                    - for the result filenames.
        """
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        super(BertExperimentInputSerializerIterationHandler, self).__init__()

        self.__sample_rows_provider = sample_rows_provider
        self.__balance_train_samples = balance_train_samples
        self.__exp_io = exp_io
        self.__exp_ctx = exp_ctx
        self.__doc_ops = doc_ops
        self.__pipeline = pipeline
        self.__data_types = data_types
        self.__save_labels_func = save_labels_func

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))

        InputDataSerializationHelper.serialize(
            pipeline=self.__pipeline,
            exp_io=self.__exp_io,
            iter_doc_ids_func=lambda dtype: self.__doc_ops.iter_doc_ids(dtype),
            keep_labels_func=lambda dtype: self.__save_labels_func(data_type),
            balance=self.__balance_train_samples,
            data_type=data_type,
            sample_rows_provider=self.__sample_rows_provider)

    # endregion

    # region protected methods

    def on_iteration(self, iter_index):
        """ Performing data serialization for a particular iteration
        """
        for data_type in self.__data_types:
            self.__handle_iteration(data_type)

    # endregion
