from arekit.common.experiment.input.views.opinions import BaseOpinionStorageView
from arekit.common.experiment.output.views.base import BaseOutputView


class OutputToOpinionCollectionsConverter(object):

    @staticmethod
    def iter_opinion_collections(output_view, opinions_view, keep_doc_id_func, to_collection_func):
        assert(isinstance(output_view, BaseOutputView))
        assert(isinstance(opinions_view, BaseOpinionStorageView))
        assert(callable(keep_doc_id_func))
        assert(callable(to_collection_func))

        for news_id in output_view.iter_news_ids():

            if not keep_doc_id_func(news_id):
                continue

            linked_data_iter = output_view.iter_linked_opinions(news_id=news_id,
                                                                opinions_view=opinions_view)

            yield news_id, to_collection_func(linked_data_iter)
