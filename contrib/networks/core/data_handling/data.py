import collections
import logging

import numpy as np

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.input.providers.row_ids.multiple import MultipleIDProvider
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.labels.base import Label
from arekit.common.model.labeling.single import SingleLabelsHelper

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

    @classmethod
    def initialize_from_experiment(cls, experiment, config, bags_collection_type):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(config, DefaultNetworkConfig))

        # Prepare necessary data.
        instance = cls(labeled_collections={}, bags_collection={})

        # Perform writing
        if not HandledData.__check_files_existed(experiment=experiment):
            instance.__perform_writing(experiment=experiment,
                                       terms_per_context=config.TermsPerContext)

        # Perform reading
        instance.__perform_reading_and_initialization(experiment=experiment,
                                                      bags_collection_type=bags_collection_type,
                                                      config=config)

        return instance

    # region writing methods

    @staticmethod
    def __perform_writing(experiment, terms_per_context):
        """
        Perform experiment input serialization
        """
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))

        term_embedding_pairs = []

        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():

            # Create annotated collection per each type.
            experiment.NeutralAnnotator.create_collection(data_type)

            # Load parsed news collections in memory.
            parsed_news_collection = experiment.create_parsed_collection(data_type)

            # Composing input.
            term_embedding_pairs = NetworkInputEncoder.to_tsv_with_embedding_and_vocabulary(
                experiment=experiment,
                data_type=data_type,
                term_embedding_pairs=term_embedding_pairs,
                parsed_news_collection=parsed_news_collection,
                terms_per_context=terms_per_context)

        # Save embedding and related vocabulary.
        NetworkInputEncoder.compose_and_save_term_embeddings_and_vocabulary(
            experiment=experiment,
            term_embedding_pairs=term_embedding_pairs)

    @staticmethod
    def __check_files_existed(experiment):
        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():
            if not NetworkIOUtils.check_files_existance(data_type=data_type,
                                                        experiment=experiment):
                return False
        return True

    # endregion

    # region reading methods

    def __perform_reading_and_initialization(self, experiment, bags_collection_type, config):
        """
        Perform reading information from the serialized experiment inputs.
        Initializing core configuration.
        """
        # Reading embedding.
        npz_embedding_data = np.load(NetworkIOUtils.get_embedding_filepath(experiment))
        config.set_term_embedding(npz_embedding_data['arr_0'])
        logger.info("Embedding readed [size={}]".format(config.TermEmbeddingMatrix.shape))

        # Reading vocabulary
        npz_vocab_data = np.load(NetworkIOUtils.get_vocab_filepath(experiment))
        vocab = dict(npz_vocab_data['arr_0'])
        logger.info("Vocabulary readed [size={}]".format(len(vocab)))

        # Reading from serialized information
        for data_type in experiment.DocumentOperations.iter_suppoted_data_types():

            labeled_sample_row_ids = self.__read_data_type(
                data_type=data_type,
                experiment=experiment,
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

    def __read_data_type(self, data_type, experiment, bags_collection_type, vocab, config):

        samples_reader = NetworkInputSampleReader.from_tsv(
            filepath=NetworkIOUtils.get_input_sample_filepath(data_type=data_type, experiment=experiment),
            row_ids_provider=MultipleIDProvider())

        labeled_sample_row_ids = list(samples_reader.iter_labeled_sample_rows(
            label_scaler=experiment.DataIO.LabelsScaler))

        self.__labeled_collections[data_type] = LabeledCollection(labeled_sample_row_ids=labeled_sample_row_ids)

        self.__bags_collection[data_type] = bags_collection_type.from_formatted_samples(
            desc="Filling bags collection [{}]".format(data_type),
            samples_reader=samples_reader,
            bag_size=config.BagSize,
            shuffle=True,
            label_scaler=experiment.DataIO.LabelsScaler,
            create_empty_sample_func=lambda cfg: InputSample.create_empty(cfg),
            create_sample_func=lambda row: self.__create_input_sample(row=row, config=config, vocab=vocab))

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

    def __create_input_sample(self, row, config, vocab):
        """
        Creates an input for Neural Network model
        """
        assert(isinstance(row, ParsedSampleRow))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(vocab, dict))

        return InputSample.from_tsv_row(
            input_sample_id=row.SampleID,
            terms=row.Terms,
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
