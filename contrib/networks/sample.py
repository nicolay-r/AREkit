import random
from itertools import chain

import numpy as np

from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.model.sample import InputSampleBase
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.features.pointers import PointersFeature
from arekit.contrib.networks.features.sample_dist import DistanceFeatures
from arekit.contrib.networks.features.term_frame_roles import FrameRoleFeatures
from arekit.contrib.networks.features.term_indices import IndicesFeature
from arekit.contrib.networks.features.term_part_of_speech import calculate_term_pos
from arekit.contrib.networks.features.term_types import calculate_term_types
from arekit.contrib.networks.features.utils import pad_right_or_crop_inplace
from arekit.processing.pos.base import POSTagger


class InputSample(InputSampleBase):
    """
    Base sample which is a part of a Bag
    It provides a to_network_input method which
    generates an input info in an appropriate way
    """

    # It is important to name with 'I_' prefix
    I_X_INDS = "x_indices"
    I_SYN_SUBJ_INDS = "syn_subj_inds"
    I_SYN_OBJ_INDS = "syn_obj_inds"
    I_SUBJ_IND = "subj_inds"
    I_OBJ_IND = "obj_inds"
    I_SUBJ_DISTS = "subj_dist"
    I_NEAREST_SUBJ_DISTS = "nearest_subj_dist"
    I_NEAREST_OBJ_DISTS = "nearest_obj_dist"
    I_OBJ_DISTS = "obj_dist"
    I_POS_INDS = "pos_inds"
    I_TERM_TYPE = "term_type"
    I_FRAME_INDS = 'frame_inds'
    I_FRAME_SENT_ROLES = 'frame_roles_inds'

    TERM_VALUE_MISSING = -1

    # TODO: Should be -1, but now it is not supported
    FRAME_SENT_ROLES_PAD_VALUE = 0
    FRAMES_PAD_VALUE = 0
    POS_PAD_VALUE = 0
    X_PAD_VALUE = 0
    TERM_TYPE_PAD_VALUE = -1
    SYNONYMS_PAD_VALUE = 0

    def __init__(self,
                 X,
                 subj_ind,
                 obj_ind,
                 syn_subj_inds,
                 syn_obj_inds,
                 dist_from_subj,
                 dist_from_obj,
                 dist_nearest_subj,
                 dist_nearest_obj,
                 pos_indices,
                 term_type,
                 frame_indices,
                 frame_sent_roles,
                 input_sample_id,
                 shift_index_dbg=0):
        assert(isinstance(X, np.ndarray))
        assert(isinstance(subj_ind, int))
        assert(isinstance(obj_ind, int))
        assert(isinstance(syn_obj_inds, np.ndarray))
        assert(isinstance(syn_subj_inds, np.ndarray))
        assert(isinstance(dist_from_subj, np.ndarray))
        assert(isinstance(dist_from_obj, np.ndarray))
        assert(isinstance(dist_nearest_subj, np.ndarray))
        assert(isinstance(dist_nearest_obj, np.ndarray))
        assert(isinstance(pos_indices, np.ndarray))
        assert(isinstance(term_type, np.ndarray))
        assert(isinstance(frame_indices, np.ndarray))
        assert(isinstance(frame_sent_roles, np.ndarray))
        assert(isinstance(shift_index_dbg, int))

        values = [(InputSample.I_X_INDS, X),
                  (InputSample.I_SUBJ_IND, subj_ind),
                  (InputSample.I_OBJ_IND, obj_ind),
                  (InputSample.I_SYN_OBJ_INDS, syn_obj_inds),
                  (InputSample.I_SYN_SUBJ_INDS, syn_subj_inds),
                  (InputSample.I_SUBJ_DISTS, dist_from_subj),
                  (InputSample.I_OBJ_DISTS, dist_from_obj),
                  (InputSample.I_NEAREST_SUBJ_DISTS, dist_nearest_subj),
                  (InputSample.I_NEAREST_OBJ_DISTS, dist_nearest_obj),
                  (InputSample.I_POS_INDS, pos_indices),
                  (InputSample.I_FRAME_INDS, frame_indices),
                  (InputSample.I_FRAME_SENT_ROLES, frame_sent_roles),
                  (InputSample.I_TERM_TYPE, term_type)]

        super(InputSample, self).__init__(shift_index_dbg=shift_index_dbg,
                                          input_sample_id=input_sample_id,
                                          values=values)

    # region class methods

    @classmethod
    def create_empty(cls, terms_per_context, frames_per_context, synonyms_per_context):
        assert(isinstance(terms_per_context, int))
        assert(isinstance(frames_per_context, int))
        assert(isinstance(synonyms_per_context, int))

        blank_synonyms = np.zeros(synonyms_per_context)
        blank_terms = np.zeros(terms_per_context)
        blank_frames = np.full(shape=frames_per_context,
                               fill_value=cls.FRAMES_PAD_VALUE)
        return cls(X=blank_terms,
                   subj_ind=0,
                   obj_ind=1,
                   syn_subj_inds=blank_synonyms,
                   syn_obj_inds=blank_synonyms,
                   dist_from_subj=blank_terms,
                   dist_from_obj=blank_terms,
                   pos_indices=blank_terms,
                   term_type=blank_terms,
                   dist_nearest_subj=blank_terms,
                   dist_nearest_obj=blank_terms,
                   frame_sent_roles=blank_terms,
                   frame_indices=blank_frames,
                   input_sample_id=u"1")

    @classmethod
    def _generate_test(cls, config):
        assert(isinstance(config, DefaultNetworkConfig))
        blank_synonyms = np.zeros(config.SynonymsPerContext)
        blank_terms = np.random.randint(0, 3, config.TermsPerContext)
        blank_frames = np.full(shape=config.FramesPerContext,
                               fill_value=cls.FRAMES_PAD_VALUE)
        return cls(X=blank_terms,
                   subj_ind=random.randint(0, 3),
                   obj_ind=random.randint(0, 3),
                   syn_subj_inds=blank_synonyms,
                   syn_obj_inds=blank_synonyms,
                   dist_from_subj=blank_terms,
                   dist_from_obj=blank_terms,
                   pos_indices=np.random.randint(0, 5, config.TermsPerContext),
                   term_type=np.random.randint(0, 3, config.TermsPerContext),
                   dist_nearest_subj=blank_terms,
                   dist_nearest_obj=blank_terms,
                   frame_sent_roles=blank_terms,
                   frame_indices=blank_frames,
                   input_sample_id=u"1")

    @classmethod
    def __get_index_by_term(cls, term, word_vocab, is_external_vocab):

        if not is_external_vocab:
            # Since we consider that all the existed words presented in vocabulary
            # we obtain the related index without any additional checks
            return word_vocab[term]

        # In case of non-native vocabulary, we consider an additional
        # placeholed when the related term has not been found in vocabulary.
        return word_vocab[term] if term in word_vocab else cls.TERM_VALUE_MISSING

    @classmethod
    def create_from_parameters(cls,
                               input_sample_id,  # row_id
                               terms,  # list of terms, that might be found in words_vocab
                               is_external_vocab,
                               subj_ind,
                               obj_ind,
                               words_vocab,  # for indexing input (all the vocabulary, obtained from offsets.py)
                               pos_tagger,
                               terms_per_context,  # for terms_per_context, frames_per_context.
                               frames_per_context,
                               synonyms_per_context,
                               syn_subj_inds,
                               syn_obj_inds,
                               frame_inds,
                               frame_sent_roles):
        """
        Here we first need to perform indexing of terms. Therefore, mark entities, frame_variants among them.
        None parameters considered as optional.
        """
        assert(isinstance(terms, list))
        assert(isinstance(frame_inds, list))
        assert(isinstance(frame_sent_roles, list))
        assert(isinstance(words_vocab, dict))
        assert(isinstance(subj_ind, int) and 0 <= subj_ind < len(terms))
        assert(isinstance(obj_ind, int) and 0 <= obj_ind < len(terms))
        assert(isinstance(pos_tagger, POSTagger))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(frames_per_context, int))
        assert(isinstance(synonyms_per_context, int))
        assert(isinstance(syn_subj_inds, list))
        assert(isinstance(syn_obj_inds, list))
        assert(subj_ind != obj_ind)

        def shift_index(ind):
            return ind - get_start_offset()

        def get_start_offset():
            return x_feature.StartIndex

        def get_end_offset():
            return x_feature.EndIndex

        entities_set = set(chain(syn_obj_inds, syn_subj_inds))

        # Composing vectors
        x_indices = np.array([cls.__get_index_by_term(term, words_vocab, is_external_vocab) for term in terms])

        # Check an ability to create sample by analyzing required window size.
        window_size = terms_per_context
        dist_between_entities = TextOpinionHelper.calc_dist_between_text_opinion_end_indices(
            pos1_ind=subj_ind,
            pos2_ind=obj_ind)

        if not cls._check_ends_could_be_fitted_in_window(dist_between_entities, window_size):
            # In some cases we may encounter with mismatched of tpc (terms per context parameter)
            # utilized during serialization stage, and the one utilized in training process.
            # If the windows size is lower in the latter case, we need to notify in order to prevent
            # from the infinite loop.
            raise Exception("Bounds for sample_id='{sample_id}' with "
                            "positions obj={obj_ind}, subj={subj_ind} "
                            "(diff={dist}) could not be fit in window, "
                            "size of {window}".format(sample_id=input_sample_id,
                                                      obj_ind=obj_ind,
                                                      subj_ind=subj_ind,
                                                      dist=dist_between_entities,
                                                      window=window_size))

        x_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=x_indices,
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.X_PAD_VALUE)

        pos_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=calculate_term_pos(terms=terms,
                                            entity_inds_set=entities_set,
                                            pos_tagger=pos_tagger),
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.POS_PAD_VALUE)

        term_type_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=calculate_term_types(terms=terms,
                                              entity_inds_set=entities_set),
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.TERM_TYPE_PAD_VALUE)

        frame_sent_roles_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=FrameRoleFeatures.to_input(frame_inds=frame_inds,
                                                    frame_sent_roles=frame_sent_roles,
                                                    size=len(terms),
                                                    filler=cls.FRAME_SENT_ROLES_PAD_VALUE),
            e1_ind=subj_ind,
            e2_ind=obj_ind,
            expected_size=window_size,
            filler=cls.FRAME_SENT_ROLES_PAD_VALUE)

        frames_feature = PointersFeature.create_shifted_and_fit(
            original_value=frame_inds,
            start_offset=get_start_offset(),
            end_offset=get_end_offset(),
            expected_size=frames_per_context,
            filler=cls.FRAMES_PAD_VALUE)

        syn_subj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=syn_subj_inds,
            start_offset=get_start_offset(),
            end_offset=get_end_offset(),
            filler=cls.SYNONYMS_PAD_VALUE)

        syn_obj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=syn_obj_inds,
            start_offset=get_start_offset(),
            end_offset=get_end_offset(),
            filler=cls.SYNONYMS_PAD_VALUE)

        shifted_subj_ind = shift_index(subj_ind)
        shifted_obj_ind = shift_index(obj_ind)

        dist_from_subj = DistanceFeatures.distance_feature(position=shifted_subj_ind, size=terms_per_context)
        dist_from_obj = DistanceFeatures.distance_feature(position=shifted_obj_ind, size=terms_per_context)

        dist_nearest_subj = DistanceFeatures.distance_abs_nearest_feature(
            positions=syn_subj_inds_feature.ValueVector,
            size=terms_per_context)

        dist_nearest_obj = DistanceFeatures.distance_abs_nearest_feature(
            positions=syn_obj_inds_feature.ValueVector,
            size=terms_per_context)

        pad_right_or_crop_inplace(lst=syn_subj_inds_feature.ValueVector,
                                  pad_size=synonyms_per_context,
                                  filler=cls.SYNONYMS_PAD_VALUE)

        pad_right_or_crop_inplace(lst=syn_obj_inds_feature.ValueVector,
                                  pad_size=synonyms_per_context,
                                  filler=cls.SYNONYMS_PAD_VALUE)

        return cls(X=np.array(x_feature.ValueVector),
                   subj_ind=shifted_subj_ind,
                   obj_ind=shifted_obj_ind,
                   syn_subj_inds=np.array(syn_subj_inds_feature.ValueVector),
                   syn_obj_inds=np.array(syn_obj_inds_feature.ValueVector),
                   dist_from_subj=dist_from_subj,
                   dist_from_obj=dist_from_obj,
                   dist_nearest_subj=dist_nearest_subj,
                   dist_nearest_obj=dist_nearest_obj,
                   pos_indices=np.array(pos_feature.ValueVector),
                   term_type=np.array(term_type_feature.ValueVector),
                   frame_indices=np.array(frames_feature.ValueVector),
                   frame_sent_roles=np.array(frame_sent_roles_feature.ValueVector),
                   input_sample_id=input_sample_id,
                   shift_index_dbg=get_start_offset())

    # endregion

    @staticmethod
    def iter_parameters():
        for var_name in dir(InputSample):
            if not var_name.startswith('I_'):
                continue
            yield getattr(InputSample, var_name)

    # region public methods

    def save(self, filepath):
        raise NotImplementedError()

    def load(self, filepath):
        raise NotImplementedError()

    # endregion