import pandas as pd
from core.evaluation.labels import Label


def calc_recall(cmp_table,
                answers,
                label,
                answer_exist,
                how_original_column,
                comparison_column):
    # TODO. cmp_table type of DocumentCmpTable
    assert(isinstance(cmp_table, pd.DataFrame))
    assert(isinstance(answers, pd.DataFrame))
    assert(isinstance(label, Label))
    assert(isinstance(answer_exist, bool))
    assert(isinstance(how_original_column, str))
    assert(isinstance(comparison_column, str))

    total = len(cmp_table[cmp_table[how_original_column] == label.to_str()])
    if total != 0:
        return 1.0 * len(answers[(answers[comparison_column] == True)]) / total
    else:
        return 0.0 if answer_exist else 1.0


def calc_precision(correct_answers,
                   answer_exist,
                   comparison_column):
    # TODO. cmp_table type of DocumentCmpTable
    assert(isinstance(correct_answers, pd.DataFrame))
    assert(isinstance(answer_exist, bool))
    assert(isinstance(comparison_column, str))
    total = len(correct_answers)
    if total != 0:
        return 1.0 * len(correct_answers[(correct_answers[comparison_column] == True)]) / total
    else:
        return 0.0 if answer_exist else 1.0


def calc_prec_and_recall(cmp_table,
                         label,
                         opinions_exist,
                         how_original_column,
                         how_results_column,
                         comparison_column):
    # TODO. cmp_table type of DocumentCmpTable
    assert(isinstance(cmp_table, pd.DataFrame))
    assert(isinstance(opinions_exist, bool))
    assert(isinstance(label, Label))
    assert(isinstance(how_results_column, str))
    assert(isinstance(how_original_column, str))
    assert(isinstance(comparison_column, str))

    correct_answers = cmp_table[(cmp_table[how_results_column] == label.to_str())]

    p = calc_precision(correct_answers=correct_answers,
                       answer_exist=opinions_exist,
                       comparison_column=comparison_column)

    r = calc_recall(cmp_table=cmp_table,
                    answers=correct_answers,
                    label=label,
                    answer_exist=opinions_exist,
                    how_original_column=how_original_column,
                    comparison_column=comparison_column)

    assert(isinstance(p, float))
    assert(isinstance(r, float))

    return p, r
