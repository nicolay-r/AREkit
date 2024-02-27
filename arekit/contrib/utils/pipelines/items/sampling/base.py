from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.utils.serializer import InputDataSerializationHelper


class BaseSerializerPipelineItem(BasePipelineItem):

    def __init__(self, rows_provider, samples_io, save_labels_func, storage, **kwargs):
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
        assert(isinstance(storage, BaseRowsStorage))
        super(BaseSerializerPipelineItem, self).__init__(**kwargs)

        self._rows_provider = rows_provider
        self._samples_io = samples_io
        self._save_labels_func = save_labels_func
        self._storage = storage

    def _serialize_iteration(self, data_type, pipeline, data_folding, doc_ids):
        assert(isinstance(data_type, DataType))
        assert(isinstance(pipeline, list))
        assert(isinstance(data_folding, dict) or data_folding is None)
        assert(isinstance(doc_ids, list) or doc_ids is None)
        assert(doc_ids is not None or data_folding is not None)

        repos = {
            "sample": InputDataSerializationHelper.create_samples_repo(
                keep_labels=self._save_labels_func(data_type),
                rows_provider=self._rows_provider,
                storage=self._storage),
        }

        writer_and_targets = {
            "sample": (self._samples_io.Writer,
                       self._samples_io.create_target(data_type=data_type)),
        }

        for description, repo in repos.items():

            if data_folding is None:
                # Consider only the predefined doc_ids.
                doc_ids_iter = doc_ids
            else:
                # Take particular data_type.
                doc_ids_iter = data_folding[data_type]
                # Consider only predefined doc_ids.
                if doc_ids is not None:
                    doc_ids_iter = set(doc_ids_iter).intersection(doc_ids)

            InputDataSerializationHelper.fill_and_write(
                repo=repo,
                pipeline=pipeline,
                doc_ids_iter=doc_ids_iter,
                desc="{desc} [{data_type}]".format(desc=description, data_type=data_type),
                writer=writer_and_targets[description][0],
                target=writer_and_targets[description][1])

    def _handle_iteration(self, data_type_pipelines, data_folding, doc_ids):
        """ Performing data serialization for a particular iteration
        """
        assert(isinstance(data_type_pipelines, dict))
        for data_type, pipeline in data_type_pipelines.items():
            self._serialize_iteration(data_type=data_type, pipeline=pipeline, data_folding=data_folding,
                                      doc_ids=doc_ids)

    def apply_core(self, input_data, pipeline_ctx):
        """
            data_type_pipelines: dict of, for example:
                {
                    DataType.Train: BasePipeline,
                    DataType.Test: BasePipeline
                }

                data_type_pipelines: doc_id -> parsed_doc -> annot -> opinion linkages
                    for example, function: sentiment_attitude_extraction_default_pipeline
                doc_ids: optional
                    this parameter allows to limit amount of documents considered for sampling
        """
        assert("data_type_pipelines" in pipeline_ctx)
        self._handle_iteration(data_type_pipelines=pipeline_ctx.provide("data_type_pipelines"),
                               doc_ids=pipeline_ctx.provide_or_none("doc_ids"),
                               data_folding=pipeline_ctx.provide_or_none("data_folding"))
