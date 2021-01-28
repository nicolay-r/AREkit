from arekit.common.evaluation.evaluators.cmp_table import DocumentCompareTable
from arekit.common.labels.base import Label


def calc_recall(cmp_table,
                answers,
                label,
                answer_exist):
    assert(isinstance(cmp_table, DocumentCompareTable))
    assert(isinstance(answers, DocumentCompareTable))
    assert(isinstance(label, Label))
    assert(isinstance(answer_exist, bool))

    total = len(cmp_table.filter_original_column_by_label(label))
    if total != 0:
        return 1.0 * len(answers.filter_comparison_true()) / total
    else:
        return 0.0 if answer_exist else 1.0


def calc_precision(correct_answers,
                   answer_exist):
    assert(isinstance(correct_answers, DocumentCompareTable))
    assert(isinstance(answer_exist, bool))

    total = len(correct_answers)
    if total != 0:
        return 1.0 * len(correct_answers.filter_comparison_true()) / total
    else:
        return 0.0 if answer_exist else 1.0


def calc_prec_and_recall(cmp_table,
                         label,
                         opinions_exist):
    assert(isinstance(cmp_table, DocumentCompareTable))
    assert(isinstance(opinions_exist, bool))
    assert(isinstance(label, Label))

    correct_answers = cmp_table.filter_result_column_by_label(label)

    p = calc_precision(correct_answers=correct_answers,
                       answer_exist=opinions_exist)

    r = calc_recall(cmp_table=cmp_table,
                    answers=correct_answers,
                    label=label,
                    answer_exist=opinions_exist)

    assert(isinstance(p, float))
    assert(isinstance(r, float))

    return p, r
