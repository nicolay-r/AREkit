import collections

from arekit.common.bound import Bound


class BaseEntitiesParser(object):

    def parse(self, sentence):
        raise NotImplementedError()

    @staticmethod
    def iter_text_with_substitutions(text, iter_subs):
        """
        text: list or string
            where we perform substitutions;
            list -- list of terms
            string -- list of chars
        iter_subs: Iterable of (value, Bound) pairs

        NOTE: substitutions should be ordered!
        """
        assert(isinstance(text, list) or isinstance(text, unicode))
        assert(isinstance(iter_subs, collections.Iterable))

        start = 0

        is_list = False
        if isinstance(text, list):
            is_list = True

        for value, bound in iter_subs:
            assert(isinstance(bound, Bound))
            assert(bound.Position >= start)

            for part in BaseEntitiesParser.__iter_text_part(text_part=text[start:bound.Position], is_list=is_list):
                yield part

            yield value

            start = bound.Position + bound.Length

        for part in BaseEntitiesParser.__iter_text_part(text_part=text[start:len(text)], is_list=is_list):
            yield part

    @staticmethod
    def __iter_text_part(text_part, is_list):
        if is_list:
            for word in text_part:
                yield word
        else:
            yield text_part
