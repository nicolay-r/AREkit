import collections
import numpy as np
from collections import OrderedDict

from core.common.frames.collection import FramesCollection
from core.common.frames.polarity import FramePolarity
from core.common.parsed_news.base import ParsedNews
from core.common.synonyms import SynonymsCollection
from core.common.labels.base import NeutralLabel
from core.networks.context.embedding import indices
from core.networks.context.configurations.base import DefaultNetworkConfig

from core.common.text_frame_variant import TextFrameVariant
from core.common.entities.base import Entity
from core.common.text_opinions.end_type import EntityEndType
from core.common.text_opinions.helper import TextOpinionHelper
from core.common.text_opinions.base import TextOpinion


class InputSample(object):
    """
    Base sample which is a part of a Bag
    It provides a to_network_input method which
    generates an input info in an appropriate way
    """

    # It is important to name with 'I_' prefix
    I_X_INDS = "x_indices"
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
    FRAME_ROLES_PAD_VALUE = 0
    FRAMES_PAD_VALUE = 0
    POS_PAD_VALUE = 0
    X_PAD_VALUE = 0
    TERM_TYPE_PAD_VALUE = -1

    def __init__(self, X,
                 subj_ind,
                 obj_ind,
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
        blank_terms = np.zeros(config.TermsPerContext)
        blank_frames = np.full(shape=config.FramesPerContext,
                               fill_value=cls.FRAMES_PAD_VALUE)
        return cls(X=blank_terms,
                   subj_ind=0,
                   obj_ind=1,
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

        sentence_index = TextOpinionHelper.extract_entity_sentence_index(text_opinion=text_opinion,
                                                                         end_type=EntityEndType.Source)

        terms = list(parsed_news.iter_sentence_terms(sentence_index))

        subj_ind = TextOpinionHelper.extract_entity_sentence_level_term_index(text_opinion=text_opinion,
                                                                              end_type=EntityEndType.Source)

        obj_ind = TextOpinionHelper.extract_entity_sentence_level_term_index(text_opinion=text_opinion,
                                                                             end_type=EntityEndType.Target)

        syn_subj_inds = TextOpinionHelper.extract_entity_sentence_level_synonym_indices(text_opinion=text_opinion,
                                                                                        end_type=EntityEndType.Source,
                                                                                        synonyms=synonyms_collection)

        syn_obj_inds = TextOpinionHelper.extract_entity_sentence_level_synonym_indices(text_opinion=text_opinion,
                                                                                       end_type=EntityEndType.Target,
                                                                                       synonyms=synonyms_collection)

        frame_inds = [index for index, _ in TextOpinionHelper.iter_frame_variants_with_indices_in_sentence(text_opinion)]

        pos_indices = indices.calculate_pos_indices_for_terms(
            terms=terms,
            pos_tagger=config.PosTagger)

        x_indices = indices.calculate_embedding_indices_for_terms(
            terms=terms,
            syn_subj_indices=set(syn_subj_inds),
            syn_obj_indices=set(syn_obj_inds),
            term_embedding_matrix=config.TermEmbeddingMatrix,
            word_embedding=config.WordEmbedding,
            custom_word_embedding=config.CustomWordEmbedding,
            token_embedding=config.TokenEmbedding,
            frames_embedding=config.FrameEmbedding)

        sentence_len = len(x_indices)

        frame_sent_roles = cls.__compose_frame_roles(
            text_opinion=text_opinion,
            size=sentence_len,
            frames_collection=frames_collection)

        term_type = InputSample.__create_term_types(terms)

        pad_size = config.TermsPerContext

        if sentence_len < pad_size:
            cls.__pad_right_inplace(frame_sent_roles, pad_size=pad_size, filler=cls.FRAME_ROLES_PAD_VALUE)
            cls.__pad_right_inplace(pos_indices, pad_size=pad_size, filler=cls.POS_PAD_VALUE)
            cls.__pad_right_inplace(x_indices, pad_size=pad_size, filler=cls.X_PAD_VALUE)
            # TODO. Provide it correct.
            cls.__pad_right_inplace(term_type, pad_size=pad_size, filler=cls.TERM_TYPE_PAD_VALUE)
        else:
            b, e, subj_ind, obj_ind = cls.__crop_bounds(
                sentence_len=sentence_len,
                window_size=config.TermsPerContext,
                e1=subj_ind,
                e2=obj_ind)

            frame_inds = cls.__shift_text_pointers(begin=b, end=e, inds=frame_inds, pad_value=cls.FRAMES_PAD_VALUE)
            syn_subj_inds = cls.__shift_text_pointers(begin=b, end=e, inds=syn_subj_inds, pad_value=0)
            syn_obj_inds = cls.__shift_text_pointers(begin=b, end=e, inds=syn_obj_inds, pad_value=0)

            cls.__crop_inplace([x_indices, frame_sent_roles, pos_indices, term_type], begin=b, end=e)

        cls.__fit_frames_dependent_indices_inplace(inds=frame_inds, frames_per_context=config.FramesPerContext)

        assert(len(frame_sent_roles) ==
               len(pos_indices) ==
               len(x_indices) ==
               len(term_type) ==
               config.TermsPerContext)

        dist_from_subj = InputSample.__dist(pos=subj_ind, size=config.TermsPerContext)
        dist_from_obj = InputSample.__dist(pos=obj_ind, size=config.TermsPerContext)
        dist_nearest_subj = InputSample.__dist_abs_nearest(positions=syn_subj_inds, size=config.TermsPerContext)
        dist_nearest_obj = InputSample.__dist_abs_nearest(positions=syn_obj_inds, size=config.TermsPerContext)

        return cls(X=np.array(x_indices),
                   subj_ind=subj_ind,
                   obj_ind=obj_ind,
                   dist_from_subj=dist_from_subj,
                   dist_from_obj=dist_from_obj,
                   dist_nearest_subj=dist_nearest_subj,
                   dist_nearest_obj=dist_nearest_obj,
                   pos_indices=np.array(pos_indices),
                   term_type=np.array(term_type),
                   frame_indices=np.array(frame_inds),
                   frame_sent_roles=np.array(frame_sent_roles),
                   text_opinion_id=text_opinion.TextOpinionID)

    # endregion

    # region private methods

    @staticmethod
    def __compose_frame_roles(text_opinion, size, frames_collection):

        result = [InputSample.FRAME_ROLES_PAD_VALUE] * size

        for index, variant in TextOpinionHelper.iter_frame_variants_with_indices_in_sentence(text_opinion):

            if index >= len(result):
                continue

            value = InputSample.__extract_uint_frame_variant_sentiment_role(
                text_frame_variant=variant,
                frames_collection=frames_collection)

            result[index] = value

        return result

    @staticmethod
    def __fit_frames_dependent_indices_inplace(inds, frames_per_context):
        if len(inds) < frames_per_context:
            InputSample.__pad_right_inplace(lst=inds,
                                            pad_size=frames_per_context,
                                            filler=InputSample.FRAMES_PAD_VALUE)
        else:
            del inds[frames_per_context:]

    @staticmethod
    def __shift_text_pointers(inds, begin, end, pad_value):
        return map(lambda frame_index: InputSample.__shift_index(w_b=begin, w_e=end,
                                                                 frame_index=frame_index,
                                                                 placeholder=pad_value),
                   inds)

    @staticmethod
    def __extract_uint_frame_variant_sentiment_role(text_frame_variant, frames_collection):
        assert(isinstance(text_frame_variant, TextFrameVariant))
        assert(isinstance(frames_collection, FramesCollection))
        frame_id = text_frame_variant.Variant.FrameID
        polarity = frames_collection.try_get_frame_sentiment_polarity(frame_id)
        if polarity is None:
            return NeutralLabel().to_uint()

        assert(isinstance(polarity, FramePolarity))

        return polarity.Label.to_uint()

    @staticmethod
    def __dist(pos, size):
        result = np.zeros(size)
        for i in xrange(len(result)):
            result[i] = i-pos if i-pos >= 0 else i-pos+size
        return result

    @staticmethod
    def __dist_abs_nearest(positions, size):
        result = np.zeros(size)
        for i in xrange(len(result)):
            result[i] = min([abs(i - p) for p in positions])
        return result

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

    @staticmethod
    def __crop_inplace(lists, begin, end):
        for i, lst in enumerate(lists):
            if end < len(lst):
                del lst[end:]
            del lst[:begin]

    @staticmethod
    def __crop_bounds(sentence_len, window_size, e1, e2):
        assert(isinstance(sentence_len, int))
        assert(isinstance(window_size, int) and window_size > 0)
        assert(isinstance(e1, int) and isinstance(e2, int))
        assert(e1 >= 0 and e2 >= 0)
        assert(e1 < sentence_len and e2 < sentence_len)
        w_begin = 0
        w_end = window_size
        while not (InputSample.__in_window(w_b=w_begin, w_e=w_end, i=e1) and
                   InputSample.__in_window(w_b=w_begin, w_e=w_end, i=e2)):
            w_begin += 1
            w_end += 1

        return w_begin, w_end, e1 - w_begin, e2 - w_begin

    @staticmethod
    def __in_window(w_b, w_e, i):
        return i >= w_b and i < w_e

    @staticmethod
    def __pad_right_inplace(lst, pad_size, filler):
        """
        Pad list ('lst') with additional elements (filler)

        lst: list
        pad_size: int
            result size
        filler: int
        returns: None
            inplace
        """
        assert(pad_size - len(lst) > 0)
        lst.extend([filler] * (pad_size - len(lst)))

    @staticmethod
    def __shift_index(w_b, w_e, frame_index, placeholder):
        shifted = frame_index - w_b
        return placeholder if not InputSample.__in_window(w_b=w_b, w_e=w_e, i=frame_index) else shifted

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
