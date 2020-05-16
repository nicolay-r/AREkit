import collections
import random

import numpy as np

from arekit.common.entities.base import Entity
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.frames.collection import FramesCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.features.dist import DistanceFeatures
from arekit.contrib.networks.features.frames import FrameFeatures
from arekit.contrib.networks.features.inds import IndicesFeature
from arekit.contrib.networks.features.pointers import PointersFeature
from arekit.contrib.networks.features.utils import pad_right_or_crop_inplace

from arekit.networks.embedding import indices
from arekit.common.model.sample import InputSampleBase


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

        super(InputSample, self).__init__(text_opinion_id=text_opinion_id,
                                          values=values)

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
                   text_opinion_id=-1)

    @classmethod
    def from_text_opinion(cls, text_opinion, parsed_news, frames_collection, synonyms_collection, config, label_scaler):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(frames_collection, FramesCollection))
        assert(isinstance(synonyms_collection, SynonymsCollection))
        assert(isinstance(label_scaler, BaseLabelScaler))

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

        frame_sent_roles = FrameFeatures.compose_frame_roles(
            text_opinion=text_opinion,
            size=len(x_indices),
            frames_collection=frames_collection,
            filler=cls.FRAME_SENT_ROLES_PAD_VALUE,
            label_scaler=label_scaler)

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
            original_value=FrameFeatures.compose_frames(text_opinion),
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

        dist_from_subj = DistanceFeatures.distance_feature(position=subj_ind, size=config.TermsPerContext)

        dist_from_obj = DistanceFeatures.distance_feature(position=obj_ind, size=config.TermsPerContext)

        dist_nearest_subj = DistanceFeatures.distance_abs_nearest_feature(
            positions=syn_subj_inds_feature.ValueVector,
            size=config.TermsPerContext)

        dist_nearest_obj = DistanceFeatures.distance_abs_nearest_feature(
            positions=syn_obj_inds_feature.ValueVector,
            size=config.TermsPerContext)

        pad_right_or_crop_inplace(lst=syn_subj_inds_feature.ValueVector,
                                  pad_size=config.SynonymsPerContext,
                                  filler=cls.SYNONYMS_PAD_VALUE)

        pad_right_or_crop_inplace(lst=syn_obj_inds_feature.ValueVector,
                                  pad_size=config.SynonymsPerContext,
                                  filler=cls.SYNONYMS_PAD_VALUE)

        return cls(X=np.array(x_feature.ValueVector),
                   subj_ind=subj_ind,
                   obj_ind=obj_ind,
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