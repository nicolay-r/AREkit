import collections
import numpy as np
from collections import OrderedDict

from arekit.common.frames.collection import FramesCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.synonyms import SynonymsCollection
from arekit.networks.context.embedding import indices
from arekit.networks.context.configurations.base import DefaultNetworkConfig

from arekit.common.entities.base import Entity
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.base import TextOpinion
from arekit.networks.context.features.dist import dist_abs_nearest_feature, distance_feature
from arekit.networks.context.features.frames import compose_frame_roles, compose_frames
from arekit.networks.context.features.inds import IndicesFeature
from arekit.networks.context.features.pointers import PointersFeature
from arekit.networks.context.features.utils import pad_right_or_crop_inplace


class InputSample(object):
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

    # TODO: Should be -1, but now it is not supported
    FRAME_SENT_ROLES_PAD_VALUE = 0
    SYNONYMS_PAD_VALUE = 0
    FRAMES_PAD_VALUE = 0
    POS_PAD_VALUE = 0
    X_PAD_VALUE = 0
    TERM_TYPE_PAD_VALUE = -1

    def __init__(self, X,
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
                 text_opinion_id):
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
        assert(isinstance(text_opinion_id, int))

        self.__text_opinion_id = text_opinion_id

        self.values = OrderedDict(
            [(InputSample.I_X_INDS, X),
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
             (InputSample.I_TERM_TYPE, term_type)])

    # region properties

    @property
    def TextOpinionID(self):
        return self.__text_opinion_id

    # endregion

    # region class methods

    @classmethod
    def create_empty(cls, config):
        assert(isinstance(config, DefaultNetworkConfig))
        blank_synonyms = np.zeros(config.SynonymsPerContext)
        blank_terms = np.zeros(config.TermsPerContext)
        blank_frames = np.full(shape=config.FramesPerContext,
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
                   text_opinion_id=-1)

    @classmethod
    def from_text_opinion(cls, text_opinion, parsed_news, frames_collection, synonyms_collection, config):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(frames_collection, FramesCollection))
        assert(isinstance(synonyms_collection, SynonymsCollection))

        sentence_index = TextOpinionHelper.extract_entity_sentence_index(
            text_opinion=text_opinion,
            end_type=EntityEndType.Source)

        terms = list(parsed_news.iter_sentence_terms(sentence_index))

        subj_ind = TextOpinionHelper.extract_entity_sentence_level_term_index(
            text_opinion=text_opinion,
            end_type=EntityEndType.Source)

        obj_ind = TextOpinionHelper.extract_entity_sentence_level_term_index(
            text_opinion=text_opinion,
            end_type=EntityEndType.Target)

        syn_subj_inds = TextOpinionHelper.extract_entity_sentence_level_synonym_indices(
            text_opinion=text_opinion,
            end_type=EntityEndType.Source,
            synonyms=synonyms_collection)

        syn_obj_inds = TextOpinionHelper.extract_entity_sentence_level_synonym_indices(
            text_opinion=text_opinion,
            end_type=EntityEndType.Target,
            synonyms=synonyms_collection)

        x_indices = indices.iter_embedding_indices_for_terms(
            terms=terms,
            syn_subj_indices=set(syn_subj_inds),
            syn_obj_indices=set(syn_obj_inds),
            term_embedding_matrix=config.TermEmbeddingMatrix,
            word_embedding=config.WordEmbedding,
            custom_word_embedding=config.CustomWordEmbedding,
            token_embedding=config.TokenEmbedding,
            frames_embedding=config.FrameEmbedding,
            use_entity_types=config.UseEntityTypesInEmbedding)

        x_indices = list(x_indices)

        frame_sent_roles = compose_frame_roles(
            text_opinion=text_opinion,
            size=len(x_indices),
            frames_collection=frames_collection,
            filler=cls.FRAME_SENT_ROLES_PAD_VALUE)

        x_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=x_indices,
            e1_in=subj_ind,
            e2_in=obj_ind,
            expected_size=config.TermsPerContext,
            filler=cls.X_PAD_VALUE)

        pos_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=indices.iter_pos_indices_for_terms(terms=terms, pos_tagger=config.PosTagger),
            e1_in=subj_ind,
            e2_in=obj_ind,
            expected_size=config.TermsPerContext,
            filler=cls.POS_PAD_VALUE)

        frame_sent_roles_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=frame_sent_roles,
            e1_in=subj_ind,
            e2_in=obj_ind,
            expected_size=config.TermsPerContext,
            filler=cls.FRAME_SENT_ROLES_PAD_VALUE)

        term_type_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=InputSample.__create_term_types(terms),
            e1_in=subj_ind,
            e2_in=obj_ind,
            expected_size=config.TermsPerContext,
            filler=cls.TERM_TYPE_PAD_VALUE)

        frames_feature = PointersFeature.create_shifted_and_fit(
            original_value=compose_frames(text_opinion),
            start_offset=x_feature.StartIndex,
            end_offset=x_feature.EndIndex,
            filler=cls.FRAMES_PAD_VALUE,
            expected_size=config.FramesPerContext)

        syn_subj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=syn_subj_inds,
            start_offset=x_feature.StartIndex,
            end_offset=x_feature.EndIndex,
            filler=0)

        syn_obj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=syn_obj_inds,
            start_offset=x_feature.StartIndex,
            end_offset=x_feature.EndIndex,
            filler=0)

        subj_ind = subj_ind - x_feature.StartIndex
        obj_ind = obj_ind - x_feature.StartIndex

        dist_from_subj = distance_feature(position=subj_ind,
                                          size=config.TermsPerContext)

        dist_from_obj = distance_feature(position=obj_ind,
                                         size=config.TermsPerContext)

        dist_nearest_subj = dist_abs_nearest_feature(positions=syn_subj_inds_feature.ValueVector,
                                                     size=config.TermsPerContext)

        dist_nearest_obj = dist_abs_nearest_feature(positions=syn_obj_inds_feature.ValueVector,
                                                    size=config.TermsPerContext)

        return cls(X=np.array(x_feature.ValueVector),
                   subj_ind=subj_ind,
                   obj_ind=obj_ind,
                   syn_subj_inds=pad_right_or_crop_inplace(lst=syn_subj_inds,
                                                           pad_size=config.SynonymsPerContext,
                                                           filler=cls.SYNONYMS_PAD_VALUE),
                   syn_obj_inds=pad_right_or_crop_inplace(lst=syn_obj_inds,
                                                          pad_size=config.SynonymsPerContext,
                                                          filler=cls.SYNONYMS_PAD_VALUE),
                   dist_from_subj=dist_from_subj,
                   dist_from_obj=dist_from_obj,
                   dist_nearest_subj=dist_nearest_subj,
                   dist_nearest_obj=dist_nearest_obj,
                   pos_indices=np.array(pos_feature.ValueVector),
                   term_type=np.array(term_type_feature.ValueVector),
                   frame_indices=np.array(frames_feature.ValueVector),
                   frame_sent_roles=np.array(frame_sent_roles_feature.ValueVector),
                   text_opinion_id=text_opinion.TextOpinionID)

    # endregion

    # region private methods

    @staticmethod
    def __create_term_types(terms):
        assert(isinstance(terms, collections.Iterable))
        feature = []
        for term in terms:
            if isinstance(term, unicode):
                feature.append(0)
            elif isinstance(term, Entity):
                feature.append(1)
            else:
                feature.append(-1)

        return feature

    # endregion

    @staticmethod
    def check_ability_to_create_sample(window_size, text_opinion):
        return abs(TextOpinionHelper.calculate_distance_between_entities_in_terms(text_opinion)) < window_size

    @staticmethod
    def iter_parameters():
        for var_name in dir(InputSample):
            if not var_name.startswith('I_'):
                continue
            yield getattr(InputSample, var_name)

    def __iter__(self):
        for key, value in self.values.iteritems():
            yield key, value

    # region public methods

    def save(self, filepath):
        raise NotImplementedError()

    def load(self, filepath):
        raise NotImplementedError()

    # endregion
