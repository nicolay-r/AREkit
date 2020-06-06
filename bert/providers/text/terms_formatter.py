from arekit.common.entities.base import Entity
from arekit.common.entities.str_mask_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType
from arekit.processing.text.token import Token


def iterate_sentence_terms(sentence_terms, entities_formatter, s_ind=None, t_ind=None):
    assert(isinstance(sentence_terms, list))
    assert(isinstance(entities_formatter, StringEntitiesFormatter))
    assert(isinstance(s_ind, int) or s_ind is None)
    assert(isinstance(t_ind, int) or t_ind is None)

    for i, term in enumerate(sentence_terms):

        if isinstance(term, unicode):
            yield term
        elif isinstance(term, Entity):
            if i == s_ind:
                yield entities_formatter.to_string(EntityType.Subject)
            elif i == t_ind:
                yield entities_formatter.to_string(EntityType.Object)
            else:
                yield entities_formatter.to_string(EntityType.Other)
        elif isinstance(term, Token):
            yield term.get_meta_value()

