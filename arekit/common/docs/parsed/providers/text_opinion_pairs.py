import logging

from arekit.common.entities.collection import EntityCollection
from arekit.common.docs.entity import DocumentEntity
from arekit.common.docs.parsed.providers.base_pairs import BasePairProvider
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.labels.provider.constant import ConstantLabelProvider

logger = logging.getLogger(__name__)


class TextOpinionPairsProvider(BasePairProvider):
    """ Document Related text opinion provider.
    """

    NAME = "text-opinion-pairs-provider"

    def __init__(self, value_to_group_id_func):
        super(TextOpinionPairsProvider, self).__init__()
        self.__value_to_group_id_func = value_to_group_id_func
        self.__doc_id = None
        self.__entities_collection = None

    @property
    def Name(self):
        return self.NAME

    def _create_pair(self, source_entity, target_entity, label):
        assert(isinstance(source_entity, DocumentEntity))
        assert(isinstance(target_entity, DocumentEntity))

        return TextOpinion(doc_id=self.__doc_id,
                           source_id=source_entity.IdInDocument,
                           target_id=target_entity.IdInDocument,
                           label=label,
                           text_opinion_id=None)

    def init_parsed_doc(self, parsed_doc):
        super(TextOpinionPairsProvider, self).init_parsed_doc(parsed_doc)
        self.__doc_id = parsed_doc.RelatedDocID
        self.__entities_collection = EntityCollection(
            entities=list(self._doc_entities),
            value_to_group_id_func=self.__value_to_group_id_func)

    def iter_from_opinion(self, opinion, debug=False):
        """ Provides text-level opinion extraction by document-level opinions
            (Opinion class instances), for a particular document (doc_id),
            with the related entity collection.
        """
        assert(isinstance(opinion, Opinion))

        key = EntityCollection.KeyType.BY_SYNONYMS
        source_entities = self.__entities_collection.try_get_entities(opinion.SourceValue, group_key=key)
        target_entities = self.__entities_collection.try_get_entities(opinion.TargetValue, group_key=key)

        if source_entities is None:
            if debug:
                logger.info("Appropriate entity for '{}'->'...' has not been found".format(
                    opinion.SourceValue))
            return
            yield

        if target_entities is None:
            if debug:
                logger.info("Appropriate entity for '...'->'{}' has not been found".format(
                    opinion.TargetValue))
            return
            yield

        label_provider = ConstantLabelProvider(label_instance=opinion.Label)

        pairs_it = self._iter_from_entities(src_entity_doc_ids=list(map(lambda e: e.IdInDocument, source_entities)),
                                            tgt_entity_doc_ids=list(map(lambda e: e.IdInDocument, target_entities)),
                                            label_provider=label_provider)

        for pair in pairs_it:
            yield pair
