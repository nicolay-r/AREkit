from arekit.common.labels.base import Label


class BaseComparator(object):

    @staticmethod
    def _cmp_result(l1, l2):
        assert (isinstance(l1, Label) or l1 is None)
        assert (isinstance(l2, Label) or l2 is None)

        if l1 is None or l2 is None:
            return False

        return l1 == l2

    def calc_diff(self, etalon, test, is_label_supported):
        raise NotImplementedError()
