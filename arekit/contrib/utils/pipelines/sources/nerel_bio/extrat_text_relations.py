from arekit.common.data.doc_provider import DocumentProvider
from arekit.common.experiment.data_type import DataType
from arekit.contrib.source.nerelbio.io_utils import NerelBioIOUtils
from arekit.contrib.source.nerelbio.versions import NerelBioVersions
from arekit.contrib.utils.pipelines.sources.nerel_bio.doc_provider import NERELBioDocProvider
from arekit.contrib.utils.pipelines.sources.nerel_bio.labels_fmt import NerelBioAnyLabelFormatter
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter


def create_text_relation_extraction_pipeline(nerel_bio_version,
                                             text_parser,
                                             label_formatter=NerelBioAnyLabelFormatter(),
                                             terms_per_context=50,
                                             doc_ops=None,
                                             docs_limit=None,
                                             custom_text_opinion_filters=None):
    assert(isinstance(nerel_bio_version, NerelBioVersions))
    assert(isinstance(doc_ops, DocumentProvider) or doc_ops is None)
    assert(isinstance(custom_text_opinion_filters, list) or custom_text_opinion_filters is None)

    data_folding = None

    if doc_ops is None:
        # Default Initialization.
        filenames_by_ids, data_folding = NerelBioIOUtils.read_dataset_split(version=nerel_bio_version,
                                                                            docs_limit=docs_limit)
        doc_ops = NERELBioDocProvider(filename_by_id=filenames_by_ids, version=nerel_bio_version)

    text_opinion_filters = [
        DistanceLimitedTextOpinionFilter(terms_per_context)
    ]

    # Append with the custom filters afterwards.
    if custom_text_opinion_filters is not None:
        text_opinion_filters += custom_text_opinion_filters

    predefined_annot = PredefinedTextOpinionAnnotator(doc_ops, label_formatter)

    pipelines = {
        DataType.Train: text_opinion_extraction_pipeline(text_parser=text_parser,
                                                         get_doc_by_id_func=doc_ops.by_id,
                                                         annotators=[predefined_annot],
                                                         entity_index_func=lambda brat_entity: brat_entity.ID,
                                                         text_opinion_filters=text_opinion_filters),
        DataType.Test: text_opinion_extraction_pipeline(text_parser=text_parser,
                                                        get_doc_by_id_func=doc_ops.by_id,
                                                        annotators=[predefined_annot],
                                                        entity_index_func=lambda brat_entity: brat_entity.ID,
                                                        text_opinion_filters=text_opinion_filters),
        DataType.Dev: text_opinion_extraction_pipeline(text_parser=text_parser,
                                                       get_doc_by_id_func=doc_ops.by_id,
                                                       annotators=[predefined_annot],
                                                       entity_index_func=lambda brat_entity: brat_entity.ID,
                                                       text_opinion_filters=text_opinion_filters),
    }

    # In the case when we setup a default data-folding.
    # There is a need to provide it, due to the needs in further.
    if data_folding is not None:
        return pipelines, data_folding

    return pipelines
