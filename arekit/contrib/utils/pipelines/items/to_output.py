from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.linkages.multilabel import MultilableOpinionLinkagesView
from arekit.common.experiment.api.io_utils import BaseIOUtils
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.model.labeling.modes import LabelCalculationMode
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.pipeline.items.handle import HandleIterPipelineItem
from arekit.contrib.utils.utils_folding import folding_iter_states, experiment_iter_index
from arekit.contrib.utils.pipelines.opinion_collections import text_opinion_linkages_to_opinion_collections_pipeline


class TextOpinionLinkagesToOpinionConverterPipelineItem(BasePipelineItem):

    def __init__(self, exp_io, create_opinion_collection_func,
                 data_type, label_scaler, labels_formatter):
        """ create_opinion_collection_func: func
                func () -> OpinionCollection (empty)
        """
        assert(isinstance(exp_io, BaseIOUtils))
        assert(callable(create_opinion_collection_func))
        assert(isinstance(data_type, DataType))
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        super(TextOpinionLinkagesToOpinionConverterPipelineItem, self).__init__()
        self._data_type = data_type

        self.__exp_io = exp_io
        self.__labels_formatter = labels_formatter
        self.__label_scaler = label_scaler
        self.__create_opinion_collection_func = create_opinion_collection_func

    def __convert(self, data_folding, output_storage, target_func):
        """ From `output_storage` to `target` conversion.
            output_storage: BaseRowsStorage
            target_func: func(doc_id) -- consdiered to provide a target for the particular document.
        """
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(output_storage, BaseRowsStorage))
        assert(callable(target_func))

        # We utilize google bert format, where every row
        # consist of label probabilities per every class
        linkages_view = MultilableOpinionLinkagesView(labels_scaler=self.__label_scaler,
                                                      storage=output_storage)

        target = self.__exp_io.create_opinions_writer_target(data_type=self._data_type,
                                                             data_folding=data_folding)

        ppl = text_opinion_linkages_to_opinion_collections_pipeline(
            iter_opinion_linkages_func=lambda doc_id: linkages_view.iter_opinion_linkages(
                doc_id=doc_id, opinions_view=self.__exp_io.create_opinions_view(target)),
            doc_ids_set=set(data_folding.fold_doc_ids_set()[self._data_type]),
            create_opinion_collection_func=self.__create_opinion_collection_func,
            labels_scaler=self.__label_scaler,
            label_calc_mode=LabelCalculationMode.AVERAGE)

        # Writing opinion collection.
        save_item = HandleIterPipelineItem(
            lambda data:
            self.__exp_io.write_opinion_collection(collection=data[1],
                                                   labels_formatter=self.__labels_formatter,
                                                   target=target_func(data[0])))

        # Executing pipeline.
        ppl.append(save_item)

        input_data = set(output_storage.iter_column_values(column_name=const.DOC_ID))

        # iterate over the result.
        for _ in ppl.run(input_data):
            pass

    def _iter_output_and_target_pairs(self, iter_index):
        raise NotImplementedError()

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("data_folding" in pipeline_ctx)

        data_folding = pipeline_ctx.provide("data_folding")
        for _ in folding_iter_states(data_folding):
            iter_index = experiment_iter_index(data_folding)
            for output_storage, target in self._iter_output_and_target_pairs(iter_index):
                self.__convert(output_storage=output_storage, target_func=target,
                               data_folding=data_folding)
