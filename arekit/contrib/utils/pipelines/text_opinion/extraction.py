from arekit.common.linkage.meta import MetaEmptyLinkedDataWrapper
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.docs.parsed.service import ParsedDocumentService
from arekit.common.docs.parser import DocumentParsers
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.pipeline.items.map import MapPipelineItem
from arekit.common.pipeline.items.map_nested import MapNestedPipelineItem
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.utils.pipelines.text_opinion.filters.base import TextOpinionFilter
from arekit.contrib.utils.pipelines.text_opinion.filters.limitation import FrameworkLimitationsTextOpinionFilter


def __iter_text_opinion_linkages(parsed_doc, annotators,
                                 is_entity_func, entity_index_func,
                                 text_opinion_filters, use_meta):
    """ use_meta: bool
            this is mainly for the progress-bar and other console parameters to stay up-to-date
            with the state in the case we do not have that much output results
            across multiple amount of documents.
    """
    assert(isinstance(annotators, list))
    assert(isinstance(parsed_doc, ParsedDocument))
    assert(isinstance(text_opinion_filters, list))
    assert(isinstance(use_meta, bool))

    def __to_id(text_opinion):
        return "{}_{}".format(text_opinion.SourceId, text_opinion.TargetId)

    service = ParsedDocumentService(parsed_doc=parsed_doc,
                                    providers=[EntityServiceProvider(entity_index_func=entity_index_func)],
                                    is_entity_func=is_entity_func)
    esp = service.get_provider(EntityServiceProvider.NAME)

    predefined = set()

    for annotator in annotators:
        for text_opinion in annotator.annotate_collection(parsed_doc=parsed_doc):
            assert(isinstance(text_opinion, TextOpinion))

            passed = True
            for f in text_opinion_filters:
                assert(isinstance(f, TextOpinionFilter))
                if not f.filter(text_opinion=text_opinion, parsed_doc=parsed_doc, entity_service_provider=esp):
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

    # This is the case to consider the end of the document.
    if use_meta:
        yield MetaEmptyLinkedDataWrapper(doc_id=parsed_doc.RelatedDocID)


def text_opinion_extraction_pipeline(pipeline_items, get_doc_by_id_func, annotators,
                                     is_entity_func, entity_index_func, batch_size,
                                     text_opinion_filters=None, use_meta_between_docs=True):
    assert(callable(get_doc_by_id_func))
    assert(callable(is_entity_func))
    assert(callable(entity_index_func))
    assert(isinstance(annotators, list))
    assert(isinstance(text_opinion_filters, list) or text_opinion_filters is None)
    assert(isinstance(use_meta_between_docs, bool))
    assert(isinstance(batch_size, int) and batch_size > 0)

    extra_filters = [] if text_opinion_filters is None else text_opinion_filters
    actual_text_opinion_filters = [FrameworkLimitationsTextOpinionFilter()] + extra_filters

    return [
        # (doc_id) -> (doc)
        MapPipelineItem(map_func=lambda doc_id: get_doc_by_id_func(doc_id)),

        # (doc, ppl_ctx) -> (parsed_doc)
        MapNestedPipelineItem(map_func=lambda doc, ppl_ctx: DocumentParsers.parse_batch(
            doc=doc, pipeline_items=pipeline_items, parent_ppl_ctx=ppl_ctx, batch_size=batch_size)),

        # (parsed_doc) -> (text_opinions)
        MapPipelineItem(map_func=lambda parsed_doc: __iter_text_opinion_linkages(
            annotators=annotators, parsed_doc=parsed_doc,
            is_entity_func=is_entity_func, entity_index_func=entity_index_func,
            text_opinion_filters=actual_text_opinion_filters, use_meta=use_meta_between_docs)),

        # linkages[] -> linkages
        FlattenIterPipelineItem()
    ]