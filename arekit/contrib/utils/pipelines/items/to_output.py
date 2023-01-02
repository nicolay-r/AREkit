from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.opinions.writer import OpinionCollectionWriter
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.pipeline.items.handle import HandleIterPipelineItem
from arekit.contrib.utils.data.views.linkages.multilabel import MultilableOpinionLinkagesView
from arekit.contrib.utils.data.views.opinions import BaseOpinionStorageView
from arekit.contrib.utils.io_utils.opinions import OpinionsIO
from arekit.contrib.utils.utils_folding import folding_iter_states, experiment_iter_index
from arekit.contrib.utils.pipelines.opinion_collections import \
    text_opinion_linkages_to_opinion_collections_pipeline_part


class TextOpinionLinkagesToOpinionConverterPipelineItem(BasePipelineItem):

    def __init__(self, opinions_io, create_opinion_collection_func,
                 opinion_collection_writer, label_scaler, labels_formatter):
        """ create_opinion_collection_func: func
                func () -> OpinionCollection (empty)
        """
        assert(isinstance(opinions_io, OpinionsIO))
        assert(callable(create_opinion_collection_func))
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(opinion_collection_writer, OpinionCollectionWriter))
        super(TextOpinionLinkagesToOpinionConverterPipelineItem, self).__init__()

        self.__opinions_io = opinions_io
        self.__labels_formatter = labels_formatter
        self.__label_scaler = label_scaler
        self.__create_opinion_collection_func = create_opinion_collection_func
        self.__opinion_collection_writer = opinion_collection_writer

    def __convert(self, data_folding, output_storage, target_func, data_type, pipeline_ctx):
        """ From `output_storage` to `target` conversion.
            output_storage: BaseRowsStorage
            target_func: func(doc_id) -- considered to provide a target for the particular document.
        """
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(output_storage, BaseRowsStorage))
        assert(isinstance(data_type, DataType))
        assert(callable(target_func))

        # We utilize google bert format, where every row
        # consist of label probabilities per every class
        linkages_view = MultilableOpinionLinkagesView(labels_scaler=self.__label_scaler,
                                                      storage=output_storage)
        target = self.__opinions_io.create_target(data_type=data_type, data_folding=data_folding)
        storage = self.__opinions_io.Reader.read(target)

        converter_part = text_opinion_linkages_to_opinion_collections_pipeline_part(
            iter_opinion_linkages_func=lambda doc_id: linkages_view.iter_opinion_linkages(
                doc_id=doc_id, opinions_view=BaseOpinionStorageView(storage)),
            doc_ids_set=set(data_folding.fold_doc_ids_set()[data_type]),
            create_opinion_collection_func=self.__create_opinion_collection_func,
            labels_scaler=self.__label_scaler,
            label_calc_mode=LabelCalculationMode.AVERAGE)

        pipeline = BasePipeline(
            converter_part +
            [HandleIterPipelineItem(lambda data: self.__opinion_collection_writer.serialize(
                collection=data[1],
                encoding='utf-8',
                labels_formatter=self.__labels_formatter,
                error_on_non_supported=True,
                target=target_func(data[0])))
             ])

        input_data = set(output_storage.iter_column_values(column_name=const.DOC_ID))

        # iterate over the result.
        for _ in pipeline.run(input_data, parent_ctx=pipeline_ctx):
            pass

    def _iter_output_and_target_pairs(self, iter_index, data_type):
        raise NotImplementedError()

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("data_folding" in pipeline_ctx)
        assert("data_type" in pipeline_ctx)

        data_folding = pipeline_ctx.provide("data_folding")
        data_type = pipeline_ctx.provide("data_type")

        for _ in folding_iter_states(data_folding):
            iter_index = experiment_iter_index(data_folding)
            pairs_it = self._iter_output_and_target_pairs(iter_index=iter_index, data_type=data_type)
            for output_storage, target in pairs_it:
                self.__convert(output_storage=output_storage,
                               target_func=target,
                               data_type=data_type,
                               data_folding=data_folding,
                               pipeline_ctx=pipeline_ctx)
