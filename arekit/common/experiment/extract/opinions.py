import collections

from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.model.labeling.base import LabelsHelper
from arekit.common.model.sample import InputSampleBase
from arekit.common.news.base import News
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection


# region private methods

def __iter_linked_text_opinion_lists(news, iter_opins_for_extraction, filter_text_opinion_func):
    assert(isinstance(news, News))
    assert(isinstance(iter_opins_for_extraction, collections.Iterable))
    assert(callable(filter_text_opinion_func))

    for opinion in iter_opins_for_extraction:
        linked_text_opinions = news.extract_linked_text_opinions(opinion)
        assert(linked_text_opinions, LinkedTextOpinionsWrapper)

        filtered_text_opinions = list(filter(filter_text_opinion_func, linked_text_opinions))

        if len(filtered_text_opinions) == 0:
            continue

        yield filtered_text_opinions

# endregions


def iter_linked_text_opins(read_news_func, news_opins_for_extraction_func,
                           parsed_news_it, terms_per_context):
    """
    Extracting text-level opinions based on doc-level opinions in documents,
    obtained by information in experiment.

    NOTE:
    1. Assumes to provide the same label (doc level opinion) onto related text-level opinions.
    """
    assert(callable(read_news_func))
    assert(isinstance(parsed_news_it, collections.Iterable))

    curr_id = 0

    for parsed_news in parsed_news_it:

        linked_text_opinion_lists = __iter_linked_text_opinion_lists(
            news=read_news_func(parsed_news.RelatedNewsID),
            iter_opins_for_extraction=news_opins_for_extraction_func(doc_id=parsed_news.RelatedNewsID),
            filter_text_opinion_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                parsed_news=parsed_news,
                text_opinion=text_opinion,
                window_size=terms_per_context))

        for linked_text_opinion_list in linked_text_opinion_lists:

            # Assign IDs.
            for text_opinion in linked_text_opinion_list:
                text_opinion.set_text_opinion_id(curr_id)
                curr_id += 1

            yield parsed_news, LinkedTextOpinionsWrapper(linked_text_opinion_list)


def fill_opinion_collection(collection, linked_data_iter, labels_helper, to_opinion_func,
                            label_calc_mode, supported_labels=None):
    """ to_opinion_func: (item, label) -> opinion
    """
    assert(isinstance(collection, OpinionCollection))
    assert(isinstance(linked_data_iter, collections.Iterable))
    assert(isinstance(labels_helper, LabelsHelper))
    assert(callable(to_opinion_func))
    assert(isinstance(supported_labels, set) or supported_labels is None)

    for linked in linked_data_iter:
        assert(isinstance(linked, LinkedDataWrapper))

        agg_label = labels_helper.aggregate_labels(
            labels_list=list(linked.iter_labels()),
            label_calc_mode=label_calc_mode)

        agg_opinion = to_opinion_func(linked.First, agg_label)

        assert(isinstance(agg_opinion, Opinion))

        if supported_labels is not None:
            if agg_opinion.Sentiment not in supported_labels:
                continue

        if collection.has_synonymous_opinion(agg_opinion):
            continue

        collection.add_opinion(agg_opinion)

    return collection
