from arekit.common.data.row_ids.base import BaseIDProvider
from arekit.common.data.views.opinions import BaseOpinionStorageView
from arekit.common.opinions.base import Opinion


def compose_opinion_by_opinion_id(ids_provider, sample_id, opinions_view, calc_label_func):
    assert(isinstance(ids_provider, BaseIDProvider))
    assert(isinstance(sample_id, str))
    assert(isinstance(opinions_view, BaseOpinionStorageView))
    assert(callable(calc_label_func))

    opinion_id = ids_provider.convert_sample_id_to_opinion_id(sample_id=sample_id)
    source, target = opinions_view.provide_opinion_info_by_opinion_id(opinion_id=opinion_id)

    return Opinion(source_value=source,
                   target_value=target,
                   sentiment=calc_label_func())


# TODO. Adopt storage.
def filter_by_id(doc_df, column, value):
    assert(isinstance(column, str))
    return doc_df[doc_df[column].str.contains(value)]
