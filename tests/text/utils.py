from arekit.common.entities.base import Entity
from arekit.processing.text.token import Token


def terms_to_str(terms):
    r = []
    for t in terms:
        if isinstance(t, Token):
            r.append(t.get_token_value())
        elif isinstance(t, Entity):
            r.append(u"[{}]".format(t.Value))
        else:
            r.append(t)
    return r
