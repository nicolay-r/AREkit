from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.news.parser import NewsParser
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.pipeline.items.map import MapPipelineItem
from arekit.common.pipeline.items.map_nested import MapNestedPipelineItem
from arekit.common.text.parser import BaseTextParser
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter
from arekit.contrib.utils.pipelines.text_opinion.filters.limitation import FrameworkLimitationsTextOpinionFilter


def __iter_text_opinion_linkages(parsed_news, annotators, text_opinion_filters):
    assert(isinstance(annotators, list))
    assert(isinstance(parsed_news, ParsedNews))
    assert(isinstance(text_opinion_filters, list))

    def __to_id(text_opinion):
        return "{}_{}".format(text_opinion.SourceId, text_opinion.TargetId)

    service = ParsedNewsService(parsed_news=parsed_news, providers=[EntityServiceProvider(None)])
    esp = service.get_provider(EntityServiceProvider.NAME)

    predefined = set()

    for annotator in annotators:
        for text_opinion in annotator.annotate_collection(parsed_news=parsed_news):
            assert(isinstance(text_opinion, TextOpinion))

            passed = True
            for f in text_opinion_filters:
                assert(isinstance(f, TextOpinionFilter))
                if not f.filter(text_opinion=text_opinion, parsed_news=parsed_news, entity_service_provider=esp):
                    passed = False
                    break

            if not passed:
                continue

            if __to_id(text_opinion) in predefined:
                # We reject those one which was already obtained
                # from the predefined sentiment annotation.
                continue

            predefined.add(__to_id(text_opinion))

            text_opinion_linkage = TextOpinionsLinkage([text_opinion])
            text_opinion_linkage.set_tag(service)
            yield text_opinion_linkage


def text_opinion_extraction_pipeline(text_parser, get_doc_by_id_func, annotators, text_opinion_filters=None):
    assert(isinstance(text_parser, BaseTextParser))
    assert(callable(get_doc_by_id_func))
    assert(isinstance(annotators, list))
    assert(isinstance(text_opinion_filters, list) or text_opinion_filters is None)

    extra_filters = [] if text_opinion_filters is None else text_opinion_filters
    actual_text_opinion_filters = [FrameworkLimitationsTextOpinionFilter()] + extra_filters

    return BasePipeline([
        # (doc_id) -> (news)
        MapPipelineItem(map_func=lambda doc_id: get_doc_by_id_func(doc_id)),

        # (news, ppl_ctx) -> (parsed_news)
        MapNestedPipelineItem(map_func=lambda news, ppl_ctx: NewsParser.parse(
            news=news, text_parser=text_parser, parent_ppl_ctx=ppl_ctx)),

        # (parsed_news) -> (text_opinions)
        MapPipelineItem(map_func=lambda parsed_news: __iter_text_opinion_linkages(
            annotators=annotators, parsed_news=parsed_news, text_opinion_filters=actual_text_opinion_filters)),

        # linkages[] -> linkages
        FlattenIterPipelineItem()
    ])
