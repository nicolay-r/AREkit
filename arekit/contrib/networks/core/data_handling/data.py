import collections
import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.input.readers.base_sample import BaseInputSampleReader
from arekit.common.experiment.labeling import LabeledCollection
from arekit.common.model.labeling.stat import calculate_labels_distribution_stat
from arekit.common.utils import check_files_existance

from arekit.contrib.networks.core.input.readers.samples_helper import NetworkInputSampleReaderHelper
from arekit.contrib.networks.core.io_utils import NetworkIOUtils
from arekit.contrib.networks.sample import InputSample
from arekit.contrib.networks.core.input.encoder import NetworkInputEncoder

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
    def check_files_existed(experiment):
        return HandledData.__check_files_existed(
            data_types_iter=experiment.DocumentOperations.DataFolding.iter_supported_data_types(),
            experiment_io=experiment.ExperimentIO)

    @staticmethod
    def serialize_from_experiment(experiment, terms_per_context, balance):
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

        HandledData.__perform_writing(experiment=experiment,
                                      terms_per_context=terms_per_context,
                                      balance=balance)

    @classmethod
    def create_empty(cls):
        return cls(labeled_collections={},
                   bags_collection={})

    def perform_reading_and_initialization(self, dtypes, exp_io, vocab,
                                           labels_count, bags_collection_type, config):
        """
        Perform reading information from the serialized experiment inputs.
        Initializing core configuration.
        """
        assert(isinstance(dtypes, collections.Iterable))
        assert(isinstance(exp_io, NetworkIOUtils))
        assert(isinstance(labels_count, int) and labels_count > 0)

        stat_uint_labeled_sample_row_ids = None

        # Reading from serialized information
        for data_type in dtypes:

            # Create samples reader.
            samples_reader = exp_io.create_samples_reader(data_type)

            # Extracting such information from serialized files.
            bags_collection, uint_labeled_sample_row_ids = self.__read_for_data_type(
                samples_reader=samples_reader,
                is_external_vocab=exp_io.has_model_predefined_state(),
                bags_collection_type=bags_collection_type,
                vocab=vocab,
                config=config,
                desc="Filling bags collection [{}]".format(data_type))

            # Saving into dictionaries.
            self.__bags_collection[data_type] = bags_collection
            self.__labeled_collections[data_type] = LabeledCollection(
                uint_labeled_sample_row_ids=uint_labeled_sample_row_ids)

            if data_type == DataType.Train:
                stat_uint_labeled_sample_row_ids = uint_labeled_sample_row_ids

        # Calculate class weights.
        if stat_uint_labeled_sample_row_ids is not None:
            normalized_label_stat, _ = calculate_labels_distribution_stat(
                uint_labeled_sample_row_ids=stat_uint_labeled_sample_row_ids,
                classes_count=labels_count)
            config.set_class_weights(normalized_label_stat)

    # region writing methods

    @staticmethod
    def __perform_writing(experiment, terms_per_context, balance):
        """
        Perform experiment input serialization
        """
        assert(isinstance(experiment, BaseExperiment))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(balance, bool))

        term_embedding_pairs = collections.OrderedDict()

        for data_type in experiment.DocumentOperations.DataFolding.iter_supported_data_types():

            # Create annotated collection per each type.
            experiment.DataIO.Annotator.serialize_missed_collections(data_type=data_type,
                                                                     doc_ops=experiment.DocumentOperations,
                                                                     opin_ops=experiment.OpinionOperations)

            # Composing input.
            NetworkInputEncoder.to_tsv_with_embedding_and_vocabulary(
                exp_io=experiment.ExperimentIO,
                exp_data=experiment.DataIO,
                opin_ops=experiment.OpinionOperations,
                doc_ops=experiment.DocumentOperations,
                entity_to_group_func=experiment.entity_to_group,
                data_type=data_type,
                term_embedding_pairs=term_embedding_pairs,
                iter_parsed_news_func=lambda: HandledData.__iter_parsed_news_func(
                    doc_ops=experiment.DocumentOperations,
                    data_type=data_type),
                terms_per_context=terms_per_context,
                balance=balance)

        # Save embedding and related vocabulary.
        NetworkInputEncoder.compose_and_save_term_embeddings_and_vocabulary(
            experiment_io=experiment.ExperimentIO,
            term_embedding_pairs=term_embedding_pairs)

    @staticmethod
    def __iter_parsed_news_func(doc_ops, data_type):
        assert(isinstance(doc_ops, DocumentOperations))
        return doc_ops.iter_parsed_news(doc_ops.iter_news_indices(data_type))

    # TODO: rename 'files' -> 'resources'
    @staticmethod
    def __check_files_existed(data_types_iter, experiment_io):
        assert(isinstance(experiment_io, NetworkIOUtils))
        for data_type in data_types_iter:

            filepaths = [
                # TODO. Samples path checking should be removed!!!
                # TODO. (We make keep them in memory and hence there is no need in the related check).
                experiment_io.get_input_sample_filepath(data_type=data_type),
                # TODO. Samples path checking should be removed!!!
                # TODO. (We make keep them in memory and hence there is no need in the related check).
                experiment_io.get_input_opinions_filepath(data_type=data_type),
                experiment_io.get_saving_vocab_filepath(),
                experiment_io.get_saving_embedding_filepath()
            ]

            if not check_files_existance(filepaths=filepaths, logger=logger):
                return False
        return True

    # endregion

    # region reading methods

    def __read_for_data_type(self, samples_reader, is_external_vocab,
                             bags_collection_type, vocab, config, desc=""):
        assert(isinstance(samples_reader, BaseInputSampleReader))

        terms_per_context = config.TermsPerContext
        frames_per_context = config.FramesPerContext
        synonyms_per_context = config.SynonymsPerContext

        bags_collection = bags_collection_type.from_formatted_samples(
            formatted_samples_iter=samples_reader.iter_rows_linked_by_text_opinions(),
            desc=desc,
            bag_size=config.BagSize,
            shuffle=True,
            create_empty_sample_func=lambda: InputSample.create_empty(
                terms_per_context=terms_per_context,
                frames_per_context=frames_per_context,
                synonyms_per_context=synonyms_per_context),
            create_sample_func=lambda row: InputSample.create_from_parameters(
                input_sample_id=row.SampleID,
                terms=row.Terms,
                entity_inds=row.EntityInds,
                is_external_vocab=is_external_vocab,
                subj_ind=row.SubjectIndex,
                obj_ind=row.ObjectIndex,
                words_vocab=vocab,
                frame_inds=row.TextFrameVariantIndices,
                frame_sent_roles=row.TextFrameVariantRoles,
                syn_obj_inds=row.SynonymObjectInds,
                syn_subj_inds=row.SynonymSubjectInds,
                terms_per_context=terms_per_context,
                frames_per_context=frames_per_context,
                synonyms_per_context=synonyms_per_context,
                pos_tags=row.PartOfSpeechTags))

        rows_it = NetworkInputSampleReaderHelper.iter_uint_labeled_sample_rows(samples_reader)

        labeled_sample_row_ids = list(rows_it)

        return bags_collection, labeled_sample_row_ids

    # endregion

    @staticmethod
    def __create_collection(data_types, collection_by_dtype_func):
        assert(isinstance(data_types, collections.Iterable))
        assert(callable(collection_by_dtype_func))

        collection = {}
        for data_type in data_types:
            collection[data_type] = collection_by_dtype_func(data_type)

        return collection
