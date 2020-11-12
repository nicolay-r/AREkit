from arekit.common.entities.base import Entity
from arekit.common.frame_variants.base import FrameVariant
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.processing.text.token import Token


def terms_to_str(terms):
    r = []
    for t in terms:
        if isinstance(t, unicode):
            r.append(t)
        elif isinstance(t, Token):
            r.append(t.get_token_value())
        elif isinstance(t, Entity):
            r.append(u"[{}]".format(t.Value))
        elif isinstance(t, TextFrameVariant):
            r.append(u"<{}>".format(t.Variant.get_value()))
        elif isinstance(t, FrameVariant):
            r.append(t.get_value())
        else:
            r.append(t)

    return r
