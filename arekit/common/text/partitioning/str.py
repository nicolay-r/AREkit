from collections.abc import Iterable

from arekit.common.bound import Bound
from arekit.common.text.partitioning.base import BasePartitioning


class StringPartitioning(BasePartitioning):
    """ NOTE: considering that provided parts
        has no intersections between each other
    """

    def provide(self, text, parts_it):
        assert(isinstance(text, str))
        assert(isinstance(parts_it, Iterable))

        start = 0
        parts = []
        for value, bound in parts_it:
            assert(isinstance(bound, Bound))
            assert(bound.Position >= start)

            # Release everything till the current value position.
            part = text[start:bound.Position]

            parts.append(part)

            # Release the entity value.
            parts.extend([value])

            start = bound.Position + bound.Length

        # Release everything after the last entity.
        last_part = text[start:len(text)]
        parts.extend([last_part])

        return parts
