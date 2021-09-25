import logging

from arekit.common.experiment.annot.base import BaseAnnotator
from arekit.common.experiment.annot.base_annot import BaseAnnotationAlgorithm
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.opinions.collection import OpinionCollection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ThreeScaleTaskAnnotator(BaseAnnotator):
    """ For three scale classification task.
    """

    def __init__(self, annot_algo):
        super(ThreeScaleTaskAnnotator, self).__init__()
        assert(isinstance(annot_algo, BaseAnnotationAlgorithm))
        self.__annot_algo = annot_algo

    @property
    def LabelsCount(self):
        return 3

    # region private methods

    def _annot_collection_core(self, parsed_news, data_type, doc_ops, opin_ops):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(data_type, DataType))
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))

        news = doc_ops.read_news(doc_id=parsed_news.RelatedNewsID)
        opinions = opin_ops.read_etalon_opinion_collection(doc_id=parsed_news.RelatedNewsID)

        annotated_opins_it = self.__annot_algo.iter_opinions(
            parsed_news=parsed_news,
            entities_collection=news.get_entities_collection(),
            existed_opinions=opinions if data_type == DataType.Train else None)

        collection = opin_ops.create_opinion_collection()
        assert(isinstance(collection, OpinionCollection))

        # Filling. Keep all the opinions without duplications.
        for opinion in annotated_opins_it:
            if collection.has_synonymous_opinion(opinion):
                continue
            collection.add_opinion(opinion)

        return collection

    # endregion

