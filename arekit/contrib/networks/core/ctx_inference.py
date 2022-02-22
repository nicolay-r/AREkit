import collections
import logging

from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.data_type import DataType
from arekit.common.model.labeling.stat import calculate_labels_distribution_stat
from arekit.contrib.networks.core.feeding.bags.collection.base import BagsCollection

from arekit.contrib.networks.core.input.rows_parser import ParsedSampleRow
from arekit.contrib.networks.sample import InputSample

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class InferenceContext(object):

    def __init__(self, sample_label_pairs_dict, bags_collections_dict):
        assert(isinstance(sample_label_pairs_dict, dict))
        assert(isinstance(bags_collections_dict, dict))
        self.__sample_label_pairs_dict = sample_label_pairs_dict
        self.__bags_collections_dict = bags_collections_dict
        self.__train_stat_uint_labeled_sample_row_ids = None

    # region Properties

    @property
    def BagsCollections(self):
        return self.__bags_collections_dict

    @property
    def SampleIdAndLabelPairs(self):
        return self.__sample_label_pairs_dict

    @property
    def HasNormalizedWeights(self):
        return self.__train_stat_uint_labeled_sample_row_ids is not None

    # endregion

    @classmethod
    def create_empty(cls):
        return cls(sample_label_pairs_dict={}, bags_collections_dict={})

    def initialize(self, dtypes, create_samples_view_func, has_model_predefined_state,
                   vocab, labels_count, bags_collection_type, bag_size, input_shapes):
        """
        Perform reading information from the serialized experiment inputs.
        Initializing core configuration.
        """
        assert(isinstance(dtypes, collections.Iterable))
        assert(callable(create_samples_view_func))
        assert(isinstance(has_model_predefined_state, bool))
        assert(isinstance(labels_count, int) and labels_count > 0)

        # Reading from serialized information
        for data_type in dtypes:

            # Create samples reader.
            samples_view = create_samples_view_func(data_type)

            # Extracting such information from serialized files.
            bags_collection = self.__read_for_data_type(
                samples_view=samples_view,
                is_external_vocab=has_model_predefined_state,
                bags_collection_type=bags_collection_type,
                vocab=vocab,
                bag_size=bag_size,
                input_shapes=input_shapes,
                desc="Filling bags collection [{}]".format(data_type))

            uint_labeled_sample_row_ids = self.__get_labeled_sample_row_ids(samples_view)

            # Saving into dictionaries.
            self.__bags_collections_dict[data_type] = bags_collection
            self.__sample_label_pairs_dict[data_type] = uint_labeled_sample_row_ids

            if data_type == DataType.Train:
                self.__train_stat_uint_labeled_sample_row_ids = uint_labeled_sample_row_ids

    def calc_normalized_weigts(self, labels_count):
        assert(isinstance(labels_count, int) and labels_count > 0)

        if self.__train_stat_uint_labeled_sample_row_ids is None:
            return

        normalized_label_stat, _ = calculate_labels_distribution_stat(
            uint_labeled_sample_row_ids=self.__train_stat_uint_labeled_sample_row_ids,
            classes_count=labels_count)

        return normalized_label_stat

    # region private methods

    @staticmethod
    def __read_for_data_type(samples_view, is_external_vocab,
                             bags_collection_type, vocab,
                             bag_size, input_shapes, desc=""):
        assert(issubclass(bags_collection_type, BagsCollection))
        assert(isinstance(samples_view, BaseSampleStorageView))

        return bags_collection_type.from_formatted_samples(
            formatted_samples_iter=samples_view.iter_rows_linked_by_text_opinions(),
            desc=desc,
            bag_size=bag_size,
            shuffle=True,
            create_empty_sample_func=lambda: InputSample.create_empty(input_shapes),
            create_sample_func=lambda row: InputSample.create_from_parameters(
                input_sample_id=row.SampleID,
                terms=row.Terms,
                entity_inds=row.EntityInds,
                is_external_vocab=is_external_vocab,
                subj_ind=row.SubjectIndex,
                obj_ind=row.ObjectIndex,
                words_vocab=vocab,
                frame_inds=row.TextFrameVariantIndices,
                frame_sent_roles=row.TextFrameConnotations,
                syn_obj_inds=row.SynonymObjectInds,
                syn_subj_inds=row.SynonymSubjectInds,
                input_shapes=input_shapes,
                pos_tags=row.PartOfSpeechTags))

    @staticmethod
    def __get_labeled_sample_row_ids(samples_view):
        rows_it = samples_view.iter_rows(handle_rows=lambda row: InferenceContext.__extract_labeled_rows(row))
        return list(rows_it)

    @staticmethod
    def __extract_labeled_rows(row):
        parsed_row = ParsedSampleRow(row)
        return parsed_row.SampleID, parsed_row.UintLabel

    # endregion
