from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.utils.serializer import InputDataSerializationHelper
from arekit.contrib.utils.utils_folding import folding_iter_states


class BaseSerializerPipelineItem(BasePipelineItem):

    def __init__(self, rows_provider, samples_io, save_labels_func, balance_func, storage):
        """ sample_rows_formatter:
                how we format input texts for a BERT model, for example:
                    - single text
                    - two sequences, separated by [SEP] token

            save_labels_func: function
                data_type -> bool
        """
        assert(isinstance(rows_provider, BaseSampleRowProvider))
        assert(isinstance(samples_io, BaseSamplesIO))
        assert(callable(save_labels_func))
        assert(callable(balance_func))
        assert(isinstance(storage, BaseRowsStorage))

        self._rows_provider = rows_provider
        self._balance_func = balance_func
        self._samples_io = samples_io
        self._save_labels_func = save_labels_func
        self._storage = storage

    def _serialize_iteration(self, data_type, pipeline, data_folding):
        assert (isinstance(data_type, DataType))
        assert (isinstance(pipeline, BasePipeline))

        repos = {
            "sample": InputDataSerializationHelper.create_samples_repo(
                keep_labels=self._save_labels_func(data_type),
                rows_provider=self._rows_provider,
                storage=self._storage),
        }

        writer_and_targets = {
            "sample": (self._samples_io.Writer,
                       self._samples_io.create_target(
                           data_type=data_type, data_folding=data_folding)),
        }

        for description, repo in repos.items():
            InputDataSerializationHelper.fill_and_write(
                repo=repo,
                pipeline=pipeline,
                doc_ids_iter=data_folding.fold_doc_ids_set()[data_type],
                do_balance=self._balance_func(data_type),
                desc="{desc} [{data_type}]".format(desc=description, data_type=data_type),
                writer=writer_and_targets[description][0],
                target=writer_and_targets[description][1])

    def _handle_iteration(self, data_type_pipelines, data_folding):
        """ Performing data serialization for a particular iteration
        """
        assert(isinstance(data_type_pipelines, dict))
        assert(isinstance(data_folding, BaseDataFolding))
        for data_type, pipeline in data_type_pipelines.items():
            self._serialize_iteration(data_type=data_type, pipeline=pipeline, data_folding=data_folding)

    def apply_core(self, input_data, pipeline_ctx):
        """
            data_type_pipelines: dict of, for example:
                {
                    DataType.Train: BasePipeline,
                    DataType.Test: BasePipeline
                }

                pipeline: doc_id -> parsed_news -> annot -> opinion linkages
                    for example, function: sentiment_attitude_extraction_default_pipeline
        """
        assert (isinstance(pipeline_ctx, PipelineContext))
        assert ("data_type_pipelines" in pipeline_ctx)
        assert ("data_folding" in pipeline_ctx)

        data_folding = pipeline_ctx.provide("data_folding")
        for _ in folding_iter_states(data_folding):
            self._handle_iteration(data_type_pipelines=pipeline_ctx.provide("data_type_pipelines"),
                                   data_folding=data_folding)
