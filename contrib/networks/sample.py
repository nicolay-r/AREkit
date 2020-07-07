import collections
import random

import numpy as np

import arekit.networks.mappers.pos
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.frames.collection import FramesCollection
from arekit.common.model.sample import InputSampleBase
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.synonyms import SynonymsCollection
from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.features.dist import DistanceFeatures
from arekit.contrib.networks.features.frames import FrameFeatures
from arekit.contrib.networks.features.inds import IndicesFeature
from arekit.contrib.networks.features.pointers import PointersFeature
from arekit.contrib.networks.features.utils import pad_right_or_crop_inplace
from arekit.networks.mappers.terms import IndexingTextTermsMapper


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
    def from_tsv_row(cls,
                     text_opinion_id,       # row_id
                     terms,                 # list of terms, that might be found in words_vocab
                     subj_ind,
                     obj_ind,
                     words_vocab,           # for indexing input (all the vocabulary, obtained from offsets.py)
                     config,                # for terms_per_context, frames_per_context.
                     frame_inds=None,
                     frame_sent_roles=None,
                     syn_subj_inds=None,
                     syn_obj_inds=None):
        """
        Here we first need to perform indexing of terms. Therefore, mark entities, frame_variants among them.
        None parameters considered as optional.
        """
        assert(isinstance(terms, list))
        assert(isinstance(frame_inds, list) or frame_inds is None)
        assert(isinstance(words_vocab, dict))
        assert(isinstance(subj_ind, int) and 0 <= subj_ind < len(terms))
        assert(isinstance(obj_ind, int) and 0 <= obj_ind < len(terms))
        assert(subj_ind != obj_ind)

        # TODO. This should be a simple mapping by embedding_matrix
        x_indices = [words_vocab[term] for term in terms]

        x_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=x_indices,
            e1_in=subj_ind,
            e2_in=obj_ind,
            expected_size=config.TermsPerContext,
            filler=cls.X_PAD_VALUE)

        pos_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=arekit.networks.mappers.pos.iter_pos_indices_for_terms(terms=terms, pos_tagger=config.PosTagger),
            e1_in=subj_ind,
            e2_in=obj_ind,
            expected_size=config.TermsPerContext,
            filler=cls.POS_PAD_VALUE)

        frame_sent_roles_feature = IndicesFeature.from_vector_to_be_fitted(
            value_vector=[cls.FRAME_SENT_ROLES_PAD_VALUE] * len(terms) if frame_sent_roles is None else [],
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
            original_value=[] if frame_inds is None else frame_inds,
            start_offset=x_feature.StartIndex,
            end_offset=x_feature.EndIndex,
            filler=cls.FRAMES_PAD_VALUE,
            expected_size=config.TermsPerContext)

        syn_subj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=[subj_ind] if syn_subj_inds is None else syn_subj_inds,
            start_offset=x_feature.StartIndex,
            end_offset=x_feature.EndIndex,
            filler=0)

        syn_obj_inds_feature = PointersFeature.create_shifted_and_fit(
            original_value=[obj_ind] if syn_obj_inds is None else syn_obj_inds,
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
                   text_opinion_id=text_opinion_id)

    # TODO. To be removed.
    # TODO. To be removed.
    # TODO. To be removed.
    @classmethod
    def from_text_opinion(cls, text_opinion, frames_collection, synonyms_collection,
                          config,
                          label_scaler,
                          string_entity_formatter,
                          text_opinion_helper):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(frames_collection, FramesCollection))
        assert(isinstance(synonyms_collection, SynonymsCollection))
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(text_opinion_helper, TextOpinionHelper))
        assert(isinstance(string_entity_formatter, StringEntitiesFormatter))

        terms = list(text_opinion_helper.iter_terms_in_related_sentence(
            text_opinion=text_opinion,
            return_ind_in_sent=False))

        subj_ind = text_opinion_helper.extract_entity_position(
            text_opinion=text_opinion,
            end_type=EntityEndType.Source,
            position_type=TermPositionTypes.IndexInSentence)

        obj_ind = text_opinion_helper.extract_entity_position(
            text_opinion=text_opinion,
            end_type=EntityEndType.Target,
            position_type=TermPositionTypes.IndexInSentence)

        subj_value = text_opinion_helper.extract_entity_value(text_opinion=text_opinion,
                                                              end_type=EntityEndType.Source)

        obj_value = text_opinion_helper.extract_entity_value(text_opinion=text_opinion,
                                                             end_type=EntityEndType.Target)

        syn_subj_inds = list(text_opinion_helper.iter_terms_in_related_sentence(
            text_opinion=text_opinion,
            return_ind_in_sent=False,
            term_check=lambda term: cls.__is_synonym_entity(term=term,
                                                            e_value=subj_value,
                                                            synonyms=synonyms_collection)))

        syn_obj_inds = list(text_opinion_helper.iter_terms_in_related_sentence(
            text_opinion=text_opinion,
            return_ind_in_sent=False,
            term_check=lambda term: cls.__is_synonym_entity(term=term,
                                                            e_value=obj_value,
                                                            synonyms=synonyms_collection)))

        # TODO. This will be removed as in Samples we deal with words and a whole vocabulary.
        # TODO. This will be removed as in Samples we deal with words and a whole vocabulary.
        # TODO. This will be removed as in Samples we deal with words and a whole vocabulary.
        term_ind_mapper = IndexingTextTermsMapper(
            syn_subj_indices=set(syn_subj_inds),
            syn_obj_indices=set(syn_obj_inds),
            term_embedding_matrix=config.TermEmbeddingMatrix,
            word_embedding=None,
            entity_embedding=None,
            token_embedding=None,
            string_entity_formatter=string_entity_formatter)

        x_indices = term_ind_mapper.iter_mapped(terms)

        x_indices = list(x_indices)

        frame_features = FrameFeatures(text_opinion_helper)

        frame_sent_roles = frame_features.compose_frame_roles(
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
            value_vector=arekit.networks.mappers.pos.iter_pos_indices_for_terms(terms=terms, pos_tagger=config.PosTagger),
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
            original_value=frame_features.compose_frames(text_opinion),
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
    def __is_synonym_entity(term, e_value, synonyms):

        if not isinstance(term, Entity):
            return False

        e_group_index = synonyms.get_synonym_group_index(e_value)

        if not synonyms.contains_synonym_value(term.Value):
            if e_value != term.Value:
                return False
        elif e_group_index != synonyms.get_synonym_group_index(term.Value):
            return False

        return True

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