import collections
import logging

import numpy as np

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.labels.base import Label
from arekit.common.model.labeling.single import SingleLabelsHelper
from arekit.common.news.parsed.collection import ParsedNewsCollection
from arekit.common.utils import check_files_existance

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.input.readers.samples import NetworkInputSampleReader
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.sample import InputSample
from arekit.contrib.networks.core.input.encoder import NetworkInputEncoder
from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HandledData(object):

    def __init__(self, labeled_collections, bags_collection):
        assert(isinstance(labeled_collections, dict))
        assert(isinstance(bags_collection, dict))
        self.__labeled_collections = labeled_collections
        self.__bags_collection = bags_collection

    # region Properties

    @property
    def BagsCollections(self):
        return self.__bags_collection

    @property
    def SamplesLabelingCollection(self):
        return self.__labeled_collections

    # endregion

    @staticmethod
    def need_serialize(experiment):
        return not HandledData.__check_files_existed(
            data_types_iter=experiment.DocumentOperations.DataFolding.iter_supported_data_types(),
            experiment_io=experiment.ExperimentIO)

    @staticmethod
    def serialize_from_experiment(experiment, terms_per_context):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))

        HandledData.__perform_writing(experiment=experiment,
                                      terms_per_context=terms_per_context)

    @classmethod
    def create_empty(cls):
        return cls(labeled_collections={},
                   bags_collection={})

    def perform_reading_and_initialization(self, experiment, bags_collection_type, config):
        """
        Perform reading information from the serialized experiment inputs.
        Initializing core configuration.
        """
        assert(isinstance(experiment, BaseExperiment))

        files_existed = HandledData.__check_files_existed(
            data_types_iter=experiment.DocumentOperations.DataFolding.iter_supported_data_types(),
            experiment_io=experiment.ExperimentIO)

        if not files_existed:
            raise Exception(u"Data has not been initialized/serialized: `{}`".format(experiment.Name))

        # Reading embedding.
        npz_embedding_data = np.load(experiment.ExperimentIO.get_loading_embedding_filepath())
        config.set_term_embedding(npz_embedding_data['arr_0'])
        logger.info("Embedding read [size={}]".format(config.TermEmbeddingMatrix.shape))

        # Reading vocabulary
        npz_vocab_data = np.load(experiment.ExperimentIO.get_loading_vocab_filepath())
        vocab = dict(npz_vocab_data['arr_0'])
        logger.info("Vocabulary read [size={}]".format(len(vocab)))

        # Reading from serialized information
        for data_type in experiment.DocumentOperations.DataFolding.iter_supported_data_types():

            labeled_sample_row_ids = self.__read_data_type(
                data_type=data_type,
                experiment_io=experiment.ExperimentIO,
                labels_scaler=experiment.DataIO.LabelsScaler,
                bags_collection_type=bags_collection_type,
                vocab=vocab,
                config=config)

            if data_type != DataType.Train:
                continue

            labels_helper = SingleLabelsHelper(label_scaler=experiment.DataIO.LabelsScaler)
            norm, _ = self.get_statistic(labeled_sample_row_ids=labeled_sample_row_ids,
                                         labels_helper=labels_helper)
            config.set_class_weights(norm)

        config.notify_initialization_completed()

    # region writing methods

    @staticmethod
    def __perform_writing(experiment, terms_per_context):
        """
        Perform experiment input serialization
        """
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))

        term_embedding_pairs = []

        for data_type in experiment.DocumentOperations.DataFolding.iter_supported_data_types():

            # Create annotated collection per each type.
            experiment.DataIO.NeutralAnnotator.serialize_missed_collections(data_type=data_type)

            # Load parsed news collections in memory.
            parsed_news_it = experiment.DocumentOperations.iter_parsed_news(
                experiment.DocumentOperations.iter_news_indices(data_type))
            parsed_news_collection = ParsedNewsCollection(parsed_news_it=parsed_news_it, notify=True)

            # Composing input.
            term_embedding_pairs = NetworkInputEncoder.to_tsv_with_embedding_and_vocabulary(
                exp_io=experiment.ExperimentIO,
                exp_data=experiment.DataIO,
                opin_ops=experiment.OpinionOperations,
                doc_ops=experiment.DocumentOperations,
                data_type=data_type,
                term_embedding_pairs=term_embedding_pairs,
                parsed_news_collection=parsed_news_collection,
                terms_per_context=terms_per_context)

        # Save embedding and related vocabulary.
        NetworkInputEncoder.compose_and_save_term_embeddings_and_vocabulary(
            experiment_io=experiment.ExperimentIO,
            term_embedding_pairs=term_embedding_pairs)

    @staticmethod
    def __check_files_existed(data_types_iter, experiment_io):
        assert(isinstance(experiment_io, NetworkIOUtils))
        for data_type in data_types_iter:

            filepaths = [
                experiment_io.get_input_sample_filepath(data_type=data_type),
                experiment_io.get_input_opinions_filepath(data_type=data_type),
                experiment_io.get_saving_vocab_filepath(),
                experiment_io.get_saving_embedding_filepath()
            ]

            if not check_files_existance(filepaths=filepaths, logger=logger):
                return False
        return True

    # endregion

    # region reading methods

    def __read_data_type(self, data_type, experiment_io, labels_scaler, bags_collection_type, vocab, config):

        samples_reader = NetworkInputSampleReader.from_tsv(
            filepath=experiment_io.get_input_sample_filepath(data_type=data_type),
            row_ids_provider=MultipleIDProvider())

        labeled_sample_row_ids = list(samples_reader.iter_labeled_sample_rows(label_scaler=labels_scaler))

        self.__labeled_collections[data_type] = LabeledCollection(labeled_sample_row_ids=labeled_sample_row_ids)

        self.__bags_collection[data_type] = bags_collection_type.from_formatted_samples(
            desc="Filling bags collection [{}]".format(data_type),
            samples_reader=samples_reader,
            bag_size=config.BagSize,
            shuffle=True,
            label_scaler=labels_scaler,
            create_empty_sample_func=lambda: InputSample.create_empty(config),
            create_sample_func=lambda row: self.__create_input_sample(
                row=row,
                config=config,
                vocab=vocab,
                is_external_vocab=experiment_io.has_model_predefined_state()))

        return labeled_sample_row_ids

    # endregion

    @staticmethod
    def get_statistic(labeled_sample_row_ids, labels_helper):
        assert(isinstance(labeled_sample_row_ids, collections.Iterable))

        stat = [0] * labels_helper.get_classes_count()

        for _, label in labeled_sample_row_ids:
            assert(isinstance(label, Label))
            stat[labels_helper.label_to_uint(label)] += 1

        total = sum(stat)
        norm = [100.0 * value / total if total > 0 else 0 for value in stat]
        return norm, stat

    @staticmethod
    def __create_input_sample(row, config, vocab, is_external_vocab):
        """
        Creates an input for Neural Network model
        """
        assert(isinstance(row, ParsedSampleRow))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(vocab, dict))

        return InputSample.from_tsv_row(
            input_sample_id=row.SampleID,
            terms=row.Terms,
            is_external_vocab=is_external_vocab,
            subj_ind=row.SubjectIndex,
            obj_ind=row.ObjectIndex,
            words_vocab=vocab,
            config=config)

    @staticmethod
    def __create_collection(data_types, collection_by_dtype_func):
        assert(isinstance(data_types, collections.Iterable))
        assert(callable(collection_by_dtype_func))

        collection = {}
        for data_type in data_types:
            collection[data_type] = collection_by_dtype_func(data_type)

        return collection
