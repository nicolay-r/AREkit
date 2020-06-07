import collections

from arekit.processing.pos.base import POSTagger
from arekit.processing.text.token import Token


# TODO. impelement as mapper to prevent from type checking.
def iter_pos_indices_for_terms(terms, pos_tagger):
    assert(isinstance(terms, collections.Iterable))
    assert(isinstance(pos_tagger, POSTagger))

    for index, term in enumerate(terms):
        if isinstance(term, Token):
            pos = pos_tagger.Empty
        elif isinstance(term, unicode):
            pos = pos_tagger.get_term_pos(term)
        else:
            pos = pos_tagger.Unknown

        yield pos_tagger.pos_to_int(pos)