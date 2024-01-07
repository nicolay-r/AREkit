from collections import Iterable

from arekit.common.bound import Bound
from arekit.common.text.partitioning.base import BasePartitioning


class Partitioning(BasePartitioning):
    """ NOTE: considering that provided parts
        has no intersections between each other
    """

    list_reg_types = {
        "str": lambda p, item: p.append(item),
        "list": lambda p, item: p.extend(item)
    }

    def __init__(self, text_fmt):
        assert(isinstance(text_fmt, str) and text_fmt in self.list_reg_types)
        self.__reg_part = self.list_reg_types[text_fmt]

    def provide(self, text, parts_it):
        assert(isinstance(parts_it, Iterable))

        parts = []
        start = 0

        for value, bound in parts_it:
            assert(isinstance(bound, Bound))
            assert(bound.Position >= start)

            # Release everything till the current value position.
            self.__reg_part(p=parts, item=text[start:bound.Position])

            # Release the entity value.
            parts.extend([value])

            start = bound.Position + bound.Length

        # Release everything after the last entity.
        self.__reg_part(p=parts, item=text[start:len(text)])

        return parts
