from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable


def calc_acc(cmp_table):
    assert(isinstance(cmp_table, DocumentCompareTable))
    cmp_table_true = cmp_table.filter_comparison_true()
    return float(len(cmp_table_true)) / (len(cmp_table) if len(cmp_table) > 0 else 1e-5)

