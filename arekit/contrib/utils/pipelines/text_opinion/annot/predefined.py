from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.base import BaseParsedNewsServiceProvider
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.opinions.converter import BratRelationConverter


class PredefinedTextOpinionAnnotator(BaseOpinionAnnotator):
    """ Brat-based text-opinion annotator (converter).
        It converts the pre-annotated Relations from BRAT-documents to TextOpinions
    """

    def __init__(self, doc_ops, label_formatter, keep_any_type=False, entity_index_func=None):
        """
            get_doc_func:
                func(doc_id)

            label_formatter: String Labels Formatter
                required for conversion.

            keep_any_type: bool
                flag that defines whether there is a need to consider all the text opinions
                or only one that supported by label formatter.

            entity_index_func: is a way of how we provide an external entity ID
                fund(entity) -> ID
        """
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(label_formatter, StringLabelsFormatter))
        assert(callable(entity_index_func) or entity_index_func is None)
        super(PredefinedTextOpinionAnnotator, self).__init__()

        self.__doc_ops = doc_ops
        self.__label_formatter = label_formatter
        self.__keep_any_type = keep_any_type
        self.__entity_index_func = (lambda brat_entity: brat_entity.ID) if \
            entity_index_func is None else entity_index_func

    @staticmethod
    def __convert_opinion_id(news, origin_id, esp):
        assert(isinstance(news, BratNews))
        assert(isinstance(origin_id, int))
        assert(isinstance(esp, BaseParsedNewsServiceProvider))

        if not news.contains_entity(origin_id):
            # Due to the complexity of entities, some entities might be nested.
            # Therefore the latter, some entities might be discarded.
            return None

        origin_entity = news.get_entity_by_id(origin_id)

        if not esp.contains_entity(origin_entity):
            return None

        document_entity = esp.get_document_entity(origin_entity)
        return document_entity.IdInDocument

    def _annot_collection_core(self, parsed_news):
        assert(isinstance(parsed_news, ParsedNews))

        pns = ParsedNewsService(parsed_news=parsed_news, providers=[
            EntityServiceProvider(self.__entity_index_func)
        ])
        esp = pns.get_provider(EntityServiceProvider.NAME)
        news = self.__doc_ops.by_id(parsed_news.RelatedDocID)

        for brat_relation in news.Relations:

            if self.__label_formatter.supports_value(brat_relation.Type) or self.__keep_any_type:

                text_opinion = BratRelationConverter.to_text_opinion(
                    brat_relation=brat_relation,
                    doc_id=parsed_news.RelatedDocID,
                    label_formatter=self.__label_formatter)

                internal_opinion = text_opinion.try_convert(
                    other=text_opinion,
                    convert_func=lambda origin_id: PredefinedTextOpinionAnnotator.__convert_opinion_id(
                        news=news, origin_id=origin_id, esp=esp))

                if internal_opinion is None:
                    continue

                yield internal_opinion
