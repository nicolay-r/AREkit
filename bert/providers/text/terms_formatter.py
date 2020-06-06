from arekit.common.entities.base import Entity
from arekit.common.entities.str_mask_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.common.synonyms import SynonymsCollection
from arekit.processing.text.token import Token


def iterate_sentence_terms(sentence_terms, entities_formatter, synonyms, s_ind=None, t_ind=None):
    assert(isinstance(sentence_terms, list))
    assert(isinstance(entities_formatter, StringEntitiesFormatter))
    assert(isinstance(synonyms, SynonymsCollection))
    assert(isinstance(s_ind, int) or s_ind is None)
    assert(isinstance(t_ind, int) or t_ind is None)

    def __syn_group(index):
        if index is None:
            return None

        entity = sentence_terms[index]
        assert (isinstance(entity, Entity))

        if not synonyms.contains_synonym_value(entity.Value):
            return None

        return synonyms.get_synonym_group_index(entity.Value)

    s_group = __syn_group(s_ind)
    t_group = __syn_group(t_ind)

    for i, term in enumerate(sentence_terms):

        if isinstance(term, unicode):
            yield term
        elif isinstance(term, Entity):
            if i == s_ind:
                yield entities_formatter.to_string(EntityType.Subject)
            elif i == t_ind:
                yield entities_formatter.to_string(EntityType.Object)
            elif __syn_group(i) == s_group:
                yield entities_formatter.to_string(EntityType.SynonymSubject)
            elif __syn_group(i) == t_group:
                yield entities_formatter.to_string(EntityType.SynonymObject)
            else:
                yield entities_formatter.to_string(EntityType.Other)
        elif isinstance(term, Token):
            yield term.get_meta_value()

