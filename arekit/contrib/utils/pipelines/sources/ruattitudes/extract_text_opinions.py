from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.labels_fmt import RuAttitudesLabelFormatter
from arekit.contrib.utils.pipelines.sources.ruattitudes.doc_provider import RuAttitudesDocumentProvider
from arekit.contrib.utils.pipelines.sources.ruattitudes.entity_filter import RuAttitudesEntityFilter
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from arekit.contrib.utils.pipelines.text_opinion.filters.entity_based import EntityBasedTextOpinionFilter


def create_text_opinion_extraction_pipeline(text_parser,
                                            label_scaler,
                                            custom_text_opinion_filters=None,
                                            version=RuAttitudesVersions.V20Large,
                                            terms_per_context=50,
                                            limit=None):
    """ Processing pipeline for RuAttitudes.
        This pipeline is based on the in-memory RuAttitudes storage.

        Original collection paper:  www.aclweb.org/anthology/r19-1118/
        Github repository:          https://github.com/nicolay-r/RuAttitudes

        version: enum
            Version of the RuAttitudes collection.
            NOTE: we consider to support a variations of the 2.0 versions.
        label_scaler:
            Scaler that allows to perform conversion from integer labels (RuAttitudes) to
            the actual `Label` instances, required in further for text_opinions instances.
        terms_per_context: int
            Amount of terms that we consider in between the Object and Subject.
        limit: int or None
            Limit of documents to consider.
    """
    assert(isinstance(label_scaler, BaseLabelScaler))
    assert(isinstance(version, RuAttitudesVersions))
    assert(isinstance(custom_text_opinion_filters, list) or custom_text_opinion_filters is None)
    assert(version in [RuAttitudesVersions.V20Large, RuAttitudesVersions.V20Base,
                       RuAttitudesVersions.V20BaseNeut, RuAttitudesVersions.V20LargeNeut])

    doc_provider = RuAttitudesDocumentProvider(version=version,
                                               keep_doc_ids_only=False,
                                               doc_id_func=lambda doc_id: doc_id,
                                               limit=limit)

    text_opinion_filters = [
        EntityBasedTextOpinionFilter(entity_filter=RuAttitudesEntityFilter()),
        DistanceLimitedTextOpinionFilter(terms_per_context)
    ]

    # Append with the custom filters afterwards.
    if custom_text_opinion_filters is not None:
        text_opinion_filters += custom_text_opinion_filters

    pipeline = text_opinion_extraction_pipeline(
        annotators=[
            PredefinedTextOpinionAnnotator(doc_provider=doc_provider,
                                           label_formatter=RuAttitudesLabelFormatter(label_scaler))
        ],
        text_opinion_filters=custom_text_opinion_filters,
        get_doc_by_id_func=doc_provider.by_id,
        entity_index_func=lambda brat_entity: brat_entity.ID,
        text_parser=text_parser)

    return pipeline
