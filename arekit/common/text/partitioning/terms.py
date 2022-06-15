import collections

from arekit.common.bound import Bound
from arekit.common.text.partitioning.base import BasePartitioning


class TermsPartitioning(BasePartitioning):
    """ NOTE: considering that provided parts
        has no intersections between each other
    """

    def provide(self, text, parts_it):
        assert(isinstance(text, list))
        assert(isinstance(parts_it, collections.Iterable))

        start = 0
        parts = []
        for value, bound in parts_it:
            assert(isinstance(bound, Bound))
            assert(bound.Position >= start)

            # Release everythig till the current value position.
            part = text[start:bound.Position]

            parts.extend(part)

            # Release the entity value.
            parts.extend([value])

            start = bound.Position + bound.Length

        # Release everything after the last entity.
        parts.extend(text[start:len(text)])

        return parts
