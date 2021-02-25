def calc_f1_single_class(prec, recall):
    if prec * recall != 0:
        return 2 * prec * recall / (prec + recall)
    else:
        return 0


def calc_f1(pos_prec, neg_prec, pos_recall, neg_recall):
    f1_pos = calc_f1_single_class(prec=pos_prec, recall=pos_recall)
    f1_neg = calc_f1_single_class(prec=neg_prec, recall=neg_recall)
    return (f1_pos + f1_neg) * 1.0 / 2


def calc_f1_3c(pos_prec, neg_prec, neu_prec, pos_recall, neg_recall, neu_recall):
    f1_pos = calc_f1_single_class(prec=pos_prec, recall=pos_recall)
    f1_neg = calc_f1_single_class(prec=neg_prec, recall=neg_recall)
    f1_neu = calc_f1_single_class(prec=neu_prec, recall=neu_recall)
    return (f1_pos + f1_neg + f1_neu) * 1.0 / 3
