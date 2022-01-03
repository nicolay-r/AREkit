import logging

from arekit.common.entities.base import Entity
from arekit.common.entities.collection import EntityCollection
from arekit.common.news.parsed.providers.base_pairs import BasePairProvider
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.labels.provider.single_label import PairSingleLabelProvider

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
        assert(isinstance(source_entity, Entity))
        assert(isinstance(target_entity, Entity))

        return TextOpinion(doc_id=self.__doc_id,
                           source_id=source_entity.IdInDocument,
                           target_id=target_entity.IdInDocument,
                           label=label,
                           owner=None,
                           text_opinion_id=None)

    def init_parsed_news(self, parsed_news):
        super(TextOpinionPairsProvider, self).init_parsed_news(parsed_news)
        self.__doc_id = parsed_news.RelatedDocID
        self.__entities_collection = EntityCollection(
            entities=list(self._entities),
            value_to_group_id_func=self.__value_to_group_id_func)

    def iter_from_opinion(self, opinion, debug=False):
        """ Provides text-level opinion extraction by document-level opinions
            (Opinion class instances), for a particular document (doc_id),
            with the related entity collection.
        """
        assert(isinstance(opinion, Opinion))

        key = self.__entities_collection.KeyType.BY_SYNONYMS
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

        label_provider = PairSingleLabelProvider(label_instance=opinion.Sentiment)

        pairs_it = self._iter_from_entities(source_entities=source_entities,
                                            target_entities=target_entities,
                                            label_provider=label_provider)

        for pair in pairs_it:
            yield pair
