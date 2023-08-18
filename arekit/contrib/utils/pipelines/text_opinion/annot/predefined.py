from arekit.common.data.doc_provider import DocumentProvider
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.docs.parsed.providers.base import BaseParsedDocumentServiceProvider
from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.docs.parsed.service import ParsedDocumentService
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.contrib.source.brat.doc import BratDocument
from arekit.contrib.source.brat.opinions.converter import BratRelationConverter


class PredefinedTextOpinionAnnotator(BaseOpinionAnnotator):
    """ Brat-based text-opinion annotator (converter).
        It converts the pre-annotated Relations from BRAT-documents to TextOpinions
    """

    def __init__(self, doc_provider, label_formatter, keep_any_type=False, entity_index_func=None):
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
        assert(isinstance(doc_provider, DocumentProvider))
        assert(isinstance(label_formatter, StringLabelsFormatter))
        assert(callable(entity_index_func) or entity_index_func is None)
        super(PredefinedTextOpinionAnnotator, self).__init__()

        self.__doc_provider = doc_provider
        self.__label_formatter = label_formatter
        self.__keep_any_type = keep_any_type
        self.__entity_index_func = (lambda brat_entity: brat_entity.ID) if \
            entity_index_func is None else entity_index_func

    @staticmethod
    def __convert_entity_id(doc, origin_entity_id, esp):
        assert(isinstance(doc, BratDocument))
        assert(isinstance(origin_entity_id, int))
        assert(isinstance(esp, BaseParsedDocumentServiceProvider))

        if not doc.contains_entity(origin_entity_id):
            # Due to the complexity of entities, some entities might be nested.
            # Therefore the latter, some entities might be discarded.
            return None

        origin_entity = doc.get_entity_by_id(origin_entity_id)

        if not esp.contains_entity(origin_entity):
            return None

        document_entity = esp.get_document_entity(origin_entity)
        return document_entity.IdInDocument

    def _annot_collection_core(self, parsed_doc):
        assert(isinstance(parsed_doc, ParsedDocument))

        pns = ParsedDocumentService(parsed_doc=parsed_doc, providers=[
            EntityServiceProvider(self.__entity_index_func)
        ])
        esp = pns.get_provider(EntityServiceProvider.NAME)
        doc = self.__doc_provider.by_id(parsed_doc.RelatedDocID)

        for brat_relation in doc.Relations:

            if self.__label_formatter.supports_value(brat_relation.Type) or self.__keep_any_type:

                text_opinion = BratRelationConverter.to_text_opinion(
                    brat_relation=brat_relation,
                    doc_id=parsed_doc.RelatedDocID,
                    label_formatter=self.__label_formatter)

                internal_opinion = text_opinion.try_convert(
                    other=text_opinion,
                    convert_entity_id_func=lambda origin_id: PredefinedTextOpinionAnnotator.__convert_entity_id(
                        doc=doc, origin_entity_id=origin_id, esp=esp))

                if internal_opinion is None:
                    continue

                yield internal_opinion
