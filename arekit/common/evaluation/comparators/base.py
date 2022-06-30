class BaseComparator(object):

    def calc_diff(self, etalon, test, is_label_supported):
        raise NotImplementedError()
