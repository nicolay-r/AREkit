from datetime import datetime

EPOCH_ARGUMENT = u"Epoch"
AVG_FIT_COST_ARGUMENT = u"avg_fit_cost"
AVG_FIT_ACC_ARGUMENT = u"avg_fit_acc"
PARAMS_SEP = u"; "
NAME_VALUE_SEP = u': '


def create_iteration_verbose_eval_msg(eval_result, data_type, epoch_index):
    title = u"Stat for [{dtype}], e={epoch}:".format(dtype=data_type, epoch=epoch_index)
    contents = [u"{doc_id}: {result}".format(doc_id=doc_id, result=result)
                for doc_id, result in eval_result.iter_document_results()]
    return u'\n'.join([title] + contents)


def create_iteration_short_eval_msg(eval_result, data_type, epoch_index, rounding_value):
    title = u"Stat for '[{dtype}]', e={epoch}".format(dtype=data_type, epoch=epoch_index)
    params = [u"{m_name}{nv_sep}{value}".format(m_name=metric_name,
                                                nv_sep=NAME_VALUE_SEP,
                                                value=round(value, rounding_value))
              for metric_name, value in eval_result.iter_total_by_param_results()]
    contents = PARAMS_SEP.join(params)
    return u'\n'.join([title, contents])


def get_message(epoch_index, avg_fit_cost, avg_fit_acc):
    """ Providing logging message
    """
    key_value_fmt = u"{k}: {v}"
    time = str(datetime.now())
    epochs = key_value_fmt.format(k=EPOCH_ARGUMENT, v=format(epoch_index))
    avg_fc = key_value_fmt.format(k=AVG_FIT_COST_ARGUMENT, v=avg_fit_cost)
    avg_ac = key_value_fmt.format(k=AVG_FIT_ACC_ARGUMENT, v=avg_fit_acc)
    return u"{time}: {epochs}: {avg_fc}, {avg_ac}".format(time=time,
                                                          epochs=epochs,
                                                          avg_fc=avg_fc,
                                                          avg_ac=avg_ac)
