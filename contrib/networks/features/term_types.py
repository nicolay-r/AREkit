import collections
from arekit.common.entities.base import Entity


def create_term_types(terms):
    assert (isinstance(terms, collections.Iterable))
    feature = []
    for term in terms:
        if isinstance(term, unicode):
            feature.append(0)
        elif isinstance(term, Entity):
            feature.append(1)
        else:
            feature.append(-1)

    return feature

