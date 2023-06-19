from arekit.common.entities.base import Entity
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.docs.parsed.term_position import TermPositionTypes, TermPosition
from arekit.common.text.enums import TermFormat
from arekit.common.text.parsed import BaseParsedText
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter


class ProfessionAsCharacteristicSentimentTextOpinionFilter(TextOpinionFilter):
    """ This is a filter, based on the PROFESSION type prefixed entity for
        the SentiNEREL collection.

        In this case, profession acts as a characteristics of the Person, and
        therefore there is no need to consider these attitudes in annotation.

        For a greater details, see:
        https://github.com/nicolay-r/AREkit/issues/404
    """

    def __init__(self, char_type="PROFESSION"):
        self.__char_type = char_type
        self.__next_entity_types = ["PERSON"]

    def filter(self, text_opinion, parsed_doc, entity_service_provider):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_doc, ParsedDocument))
        assert(isinstance(entity_service_provider, EntityServiceProvider))

        # Picking up entity.
        target_entity = entity_service_provider._doc_entities[text_opinion.TargetId]
        assert(isinstance(target_entity, Entity))

        if target_entity.Type != self.__char_type:
            # This is not our case.
            return True

        # Picking up the related target entity position.
        target_pos = entity_service_provider.get_entity_position(text_opinion.TargetId)
        assert(isinstance(target_pos, TermPosition))

        # Picking up the related sentence of target.
        t_sent = target_pos.get_index(TermPositionTypes.SentenceIndex)
        sentence = parsed_doc.get_sentence(t_sent)
        assert(isinstance(sentence, BaseParsedText))

        # Picking up the entity position in sentence.
        target_term_ind = target_pos.get_index(TermPositionTypes.IndexInSentence)

        # We pick up the next term within the parsed sentece.
        next_term = sentence.get_term(target_term_ind + 1, term_format=TermFormat.Raw) \
            if len(sentence) > target_term_ind + 1 else None

        if next_term is None:
            # This is not our case.
            return True

        if isinstance(next_term, Entity) and next_term.Type in self.__next_entity_types:
            # We reject this opinion from the annotation, since this is not expected to be a sentiment one.
            return False

        return True
