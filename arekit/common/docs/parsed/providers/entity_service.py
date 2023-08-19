from enum import Enum

from arekit.common.entities.base import Entity
from arekit.common.docs.entity import DocumentEntity
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.providers.base import BaseParsedDocumentServiceProvider
from arekit.common.docs.parsed.term_position import TermPositionTypes, TermPosition
from arekit.common.text_opinions.base import TextOpinion


class EntityEndType(Enum):
    """ Pair end type
    """
    Source = 1
    Target = 2


class DistanceType(Enum):
    InTerms = 1
    InSentences = 2

    @staticmethod
    def to_position_type(dist_type):
        assert(isinstance(dist_type, DistanceType))

        if dist_type == DistanceType.InTerms:
            return TermPositionTypes.IndexInDocument

        if dist_type == DistanceType.InSentences:
            return TermPositionTypes.SentenceIndex


class EntityServiceProvider(BaseParsedDocumentServiceProvider):
    """ This class provides a helper functions for TextOpinions, which become a part of TextOpinionCollection.
        The latter is important because of the dependency from Owner.
        We utilize 'extract' prefix in methods to emphasize that these are methods of helper.

        Wrapper over:
            parsed doc, positions, text_opinions
    """

    NAME = "entity-service-provider"

    def __init__(self, entity_index_func):
        assert(callable(entity_index_func))
        super(EntityServiceProvider, self).__init__(entity_index_func=entity_index_func)
        # Initialize API.
        self.__iter_raw_terms_func = None
        # Initialize entity positions.
        self.__entity_positions = None

    @property
    def Name(self):
        return self.NAME

    def init_parsed_doc(self, parsed_doc):
        super(EntityServiceProvider, self).init_parsed_doc(parsed_doc)
        assert(isinstance(parsed_doc, ParsedDocument))
        self.__iter_raw_terms_func = lambda: parsed_doc.iter_terms(filter_func=None, term_only=False)
        self.__entity_positions = self.__calculate_entity_positions()

    # region public 'extract' methods

    def extract_entity_value(self, text_opinion, end_type):
        return self.__extract_entity_value(text_opinion=text_opinion, end_type=end_type)

    def extract_entity_position(self, text_opinion, end_type, position_type=None):
        return self.__get_entity_position(text_opinion=text_opinion,
                                          end_type=end_type,
                                          position_type=position_type)

    # endregion

    # region public 'calculate' methods

    @staticmethod
    def calc_dist_between_text_opinion_end_indices(pos1_ind, pos2_ind):
        return EntityServiceProvider.__calc_distance_by_inds(pos1_ind=pos1_ind, pos2_ind=pos2_ind)

    def calc_dist_between_text_opinion_ends(self, text_opinion, distance_type):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(distance_type, DistanceType))

        e1_id = self.__get_end_id(text_opinion=text_opinion, end_type=EntityEndType.Source)
        e2_id = self.__get_end_id(text_opinion=text_opinion, end_type=EntityEndType.Target)

        return self.__calc_distance(
            pos1=self.get_entity_position(id_in_document=e1_id),
            pos2=self.get_entity_position(id_in_document=e2_id),
            position_type=DistanceType.to_position_type(distance_type))

    def calc_dist_between_entities(self, e1, e2, distance_type):
        assert(isinstance(e1, DocumentEntity))
        assert(isinstance(e2, DocumentEntity))
        assert(isinstance(distance_type, DistanceType))

        return self.__calc_distance(
            pos1=self.get_entity_position(e1.IdInDocument),
            pos2=self.get_entity_position(e2.IdInDocument),
            position_type=DistanceType.to_position_type(distance_type))

    def get_entity_position(self, id_in_document, position_type=None):
        """ returns: TermPosition or int
        """
        assert(isinstance(position_type, TermPositionTypes) or position_type is None)

        e_pos = self.__entity_positions[id_in_document]
        assert(isinstance(e_pos, TermPosition))

        if position_type is None:
            return e_pos

        return e_pos.get_index(position_type)

    def get_entity_value(self, id_in_document):
        entity = self._doc_entities[id_in_document]
        assert(isinstance(entity, Entity))
        return entity.Value

    # endregion

    # region private methods

    def __extract_entity_value(self, text_opinion, end_type):
        assert(isinstance(text_opinion, TextOpinion))
        end_id = self.__get_end_id(text_opinion=text_opinion, end_type=end_type)
        return self.get_entity_value(end_id)

    def __get_entity_position(self, text_opinion, end_type, position_type=None):
        assert(isinstance(text_opinion, TextOpinion))
        end_id = self.__get_end_id(text_opinion=text_opinion, end_type=end_type)
        return self.get_entity_position(end_id, position_type)

    def __calc_distance(self, pos1, pos2, position_type=TermPositionTypes.IndexInDocument):
        assert(isinstance(pos1, TermPosition))
        assert(isinstance(pos2, TermPosition))
        return self.__calc_distance_by_inds(pos1_ind=pos1.get_index(position_type),
                                            pos2_ind=pos2.get_index(position_type))

    @staticmethod
    def __calc_distance_by_inds(pos1_ind, pos2_ind):
        return abs(pos1_ind - pos2_ind)

    @staticmethod
    def __get_end_id(text_opinion, end_type):
        assert(isinstance(text_opinion, TextOpinion))
        assert(end_type == EntityEndType.Source or end_type == EntityEndType.Target)
        return text_opinion.SourceId if end_type == EntityEndType.Source else text_opinion.TargetId

    def __calculate_entity_positions(self):
        """ Note: here we consider the same order as in self._entities.
        """
        t_ind_in_doc = -1

        positions = {}
        for s_ind, t_ind_in_sent, term in self.__iter_raw_terms_func():

            t_ind_in_doc += 1

            if not isinstance(term, Entity):
                continue

            # We consider that entities within a single tree has the same positions.
            for tree_entity in list(term.iter_childs()) + [term]:

                key = self.get_document_entity(tree_entity).IdInDocument
                assert(key not in positions)

                positions[key] = TermPosition(term_ind_in_doc=t_ind_in_doc,
                                              term_ind_in_sent=t_ind_in_sent,
                                              s_ind=s_ind)

        return positions

    # endregion
