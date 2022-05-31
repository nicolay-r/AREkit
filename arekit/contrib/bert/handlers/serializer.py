from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.samplers.factory import create_bert_sample_provider
from arekit.contrib.utils.serializer import InputDataSerializationHelper


class BertExperimentInputSerializerIterationHandler(ExperimentIterationHandler):

    def __init__(self, exp_io, exp_ctx, doc_ops, opin_ops,
                 sample_labels_fmt, annot_labels_fmt, value_to_group_id_func,
                 sample_provider_type, entity_formatter, balance_train_samples,
                 text_parser):
        assert(isinstance(exp_io, BaseIOUtils))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(sample_labels_fmt, StringLabelsFormatter))
        assert(isinstance(annot_labels_fmt, StringLabelsFormatter))
        assert(isinstance(text_parser, BaseTextParser))
        assert(callable(value_to_group_id_func))
        super(BertExperimentInputSerializerIterationHandler, self).__init__()

        self.__sample_rows_provider = create_bert_sample_provider(
            text_b_labels_fmt=sample_labels_fmt,
            provider_type=sample_provider_type,
            label_scaler=self.__exp_ctx.LabelsScaler,
            entity_formatter=entity_formatter)

        self.__value_to_group_id_func = value_to_group_id_func
        self.__balance_train_samples = balance_train_samples
        self.__exp_io = exp_io
        self.__exp_ctx = exp_ctx
        self.__doc_ops = doc_ops
        self.__opin_ops = opin_ops
        self.__text_parser = text_parser

    # region private methods

    def __handle_iteration(self, data_type):
        assert(isinstance(data_type, DataType))

        InputDataSerializationHelper.serialize(
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=self.__doc_ops,
            opin_ops=self.__opin_ops,
            terms_per_context=self.__exp_ctx.TermsPerContext,
            balance=self.__balance_train_samples,
            value_to_group_id_func=self.__value_to_group_id_func,
            text_parser=self.__text_parser,
            data_type=data_type,
            sample_rows_provider=self.__sample_rows_provider)

    # endregion

    # region protected methods

    def on_iteration(self, iter_index):
        """ Performing data serialization for a particular iteration
        """
        for data_type in self.__exp_ctx.DataFolding.iter_supported_data_types():
            self.__handle_iteration(data_type)

    # endregion
