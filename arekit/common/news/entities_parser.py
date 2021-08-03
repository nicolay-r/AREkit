from arekit.common.bound import Bound
from arekit.common.news.sentence import BaseNewsSentence


class BaseEntitiesParser(object):

    def parse(self, sentence):
        """
        text: list or string
            where we perform substitutions;
            list -- list of terms
            string -- list of chars
        iter_subs: Iterable of (value, Bound) pairs

        NOTE: substitutions should be ordered!
        """
        assert(isinstance(sentence, BaseNewsSentence))

        start = 0
        result = []

        self._before_parsing(sentence)

        for value, bound in self._iter_subs_values_with_bounds():
            assert(isinstance(bound, Bound))
            assert(bound.Position >= start)

            # Release everything till the current value position.
            part = self._iter_part(from_index=start, to_index=bound.Position)
            result.extend(part)

            # Release the entity value.
            result.extend([value])

            start = bound.Position + bound.Length

        # Release everything after the last entity.
        last_part = self._iter_part(from_index=start,
                                    to_index=self._get_sentence_length())
        result.extend(last_part)

        return result

    # region protected methods

    def _before_parsing(self, sentence):
        raise NotImplementedError()

    def _get_sentence_length(self):
        raise NotImplementedError()

    def _iter_subs_values_with_bounds(self):
        """ Provides pairs (value, bound)
        """
        raise NotImplementedError()

    def _iter_part(self, from_index, to_index):
        """ Iters over parts in the following range:
            [from_index, to_index)
        """
        raise NotImplementedError()

    # endregion