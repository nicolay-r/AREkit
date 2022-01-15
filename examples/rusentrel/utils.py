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
