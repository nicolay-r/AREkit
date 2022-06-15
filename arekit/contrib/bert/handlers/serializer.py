from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.handler import ExperimentIterationHandler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.samplers.factory import create_bert_sample_provider
from arekit.contrib.utils.pipelines.annot.base import sentiment_attitude_extraction_default_pipeline
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

        # We adopt as an example the pipeline, in which
        # we provide a manual annotation for a given doc_id
        default_pipeline = sentiment_attitude_extraction_default_pipeline(
            annotator=self.__exp_ctx.Annotator,
            opin_ops=self.__opin_ops,
            get_doc_func=lambda doc_id: self.__doc_ops.get_doc(doc_id),
            terms_per_context=self.__exp_ctx.TermsPerContext,
            value_to_group_id_func=self.__value_to_group_id_func,
            data_type=data_type,
            text_parser=self.__text_parser)

        InputDataSerializationHelper.serialize(
            pipeline=default_pipeline,
            exp_io=self.__exp_io,
            iter_doc_ids_func=lambda dtype: self.__doc_ops.iter_doc_ids(dtype),
            balance=self.__balance_train_samples,
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
