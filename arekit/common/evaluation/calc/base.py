class BaseComparisonCalculator(object):

    def calc_diff(self, etalon_opins, test_opins, is_label_supported):
        raise NotImplementedError()
