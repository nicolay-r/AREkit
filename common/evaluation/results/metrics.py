from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.labels.base import Label


def calc_recall(origin_answers, result_answers, answer_exist):
    """ Calculate Recall for a particular class:
        -- How many correctly labeled (result_answers.filter() -- TP)
        items were found comparing with the class volume in
        etalon labeling (origin answers)
       Equation: TP/(TP + FN)
    """
    assert(isinstance(origin_answers, DocumentCompareTable))
    assert(isinstance(result_answers, DocumentCompareTable))
    assert(isinstance(answer_exist, bool))

    total = len(origin_answers)
    if total != 0:
        return 1.0 * len(result_answers.filter_comparison_true()) / total
    else:
        return 0.0 if answer_exist else 1.0


def calc_precision(result_answers, answer_exist):
    """ Calculate Precision for a particular class:
        -- How many selected items (result_answers) are relevant to etalon answers?
       Equation: TP/(TP + FN)
    """
    assert(isinstance(result_answers, DocumentCompareTable))
    assert(isinstance(answer_exist, bool))

    total = len(result_answers)
    if total != 0:
        return 1.0 * len(result_answers.filter_comparison_true()) / total
    else:
        return 0.0 if answer_exist else 1.0


def calc_prec_and_recall(cmp_table,
                         label,
                         opinions_exist):
    assert(isinstance(cmp_table, DocumentCompareTable))
    assert(isinstance(opinions_exist, bool))
    assert(isinstance(label, Label))

    # Extracting class-related entries from cmp_table.
    result_answers = cmp_table.filter_result_column_by_label(label)
    origin_answers = cmp_table.filter_original_column_by_label(label)

    p = calc_precision(result_answers=result_answers,
                       answer_exist=opinions_exist)

    r = calc_recall(origin_answers=origin_answers,
                    result_answers=result_answers,
                    answer_exist=opinions_exist)

    assert(isinstance(p, float))
    assert(isinstance(r, float))

    return p, r
