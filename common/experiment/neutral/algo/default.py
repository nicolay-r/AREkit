import collections

from arekit.common.entities.base import Entity
from arekit.common.entities.collection import EntityCollection
from arekit.common.experiment.neutral.algo.base import BaseNeutralAnnotationAlgorithm
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.parsed_news.term_position import TermPositionTypes
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils


class DefaultNeutralAnnotationAlgorithm(BaseNeutralAnnotationAlgorithm):
    """
    Neutral annotation algorithm which assumes to compose pairs
    within a sentence which are not a part of sentiment.
    """

    class DistanceType:
        InTerms = 'in_terms'
        InSentences = 'in_sentences'

    def __init__(self,
                 synonyms,
                 create_parsed_news_func,
                 iter_news_ids,
                 ignored_entity_values=None):
        """
        create_opinion_func:
            func (source_value, target_value, sentiment) -> Opinion
        create_opinion_collection_func:
            func () -> OpinionCollection
        """
        assert(isinstance(synonyms, SynonymsCollection))
        assert(callable(create_parsed_news_func))
        assert(isinstance(iter_news_ids, collections.Iterable))
        assert(isinstance(ignored_entity_values, list) or ignored_entity_values is None)

        self.__synonyms = synonyms
        self.__ignored_entity_values = [] if ignored_entity_values is None else ignored_entity_values

        self.__pnc = ParsedNewsCollection()
        for doc_id in RuSentRelIOUtils.iter_collection_indices():
            parsed_news = create_parsed_news_func(doc_id)
            self.__pnc.add(parsed_news)

    # region private methods

    @staticmethod
    def __create_key_by_entity_pair(e1, e2):
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))
        return u"{}_{}".format(e1.IdInDocument, e2.IdInDocument)

    def __is_ignored_entity_value(self, entity_value):
        assert(isinstance(entity_value, unicode))
        return entity_value in self.__ignored_entity_values

    # TODO: Leave single func with the param e -> index.
    def __get_distance_in_sentences_between_entities(self, n_id, e1, e2):
        """
        distance_type: string
        """
        assert(isinstance(e1, Entity))
        assert(isinstance(e2, Entity))

        parsed_news = self.__pnc.get_by_news_id(n_id)

        assert(isinstance(parsed_news, ParsedNews))

        # TODO. Utilize helper.
        # TODO. Simplify
        e1_ind = parsed_news.get_entity_position(e1.IdInDocument).get_index(position_type=TermPositionTypes.SentenceIndex)
        e2_ind = parsed_news.get_entity_position(e2.IdInDocument).get_index(position_type=TermPositionTypes.SentenceIndex)
        return abs(e1_ind - e2_ind)

    def __create_opinions_between_entities(self, relevant_pairs, entities_collection):
        assert(isinstance(entities_collection, EntityCollection))
        assert(self.__synonyms.IsReadOnly is True)

        extracted_count = 0
        neutral_opinions = OpinionCollection(opinions=None,
                                             synonyms=self.__synonyms)

        for e1 in entities_collection:
            assert(isinstance(e1, Entity))

            for e2 in entities_collection:
                assert(isinstance(e2, Entity))

                key = self.__create_key_by_entity_pair(e1=e1, e2=e2)
                if key not in relevant_pairs:
                    continue

                opinion = Opinion(source_value=e1.Value,
                                  target_value=e2.Value,
                                  sentiment=NeutralLabel())

                if neutral_opinions.has_synonymous_opinion(opinion):
                    continue

                neutral_opinions.add_opinion(opinion)
                extracted_count += 1

        print "Neutral opinions extracted: {}".format(extracted_count)

        return neutral_opinions

    # endregion

    def make_neutrals(self, news_id, entities_collection, sentiment_opinions=None):
        assert(isinstance(news_id, int))
        assert(isinstance(entities_collection, EntityCollection))

        relevant_pairs = {}

        for e1 in entities_collection:
            assert(isinstance(e1, Entity))

            for e2 in entities_collection:
                assert(isinstance(e2, Entity))

                # TODO. In separated method

                if e1.IdInDocument == e2.IdInDocument:
                    continue

                if self.__is_ignored_entity_value(entity_value=e1.Value):
                    continue
                if self.__is_ignored_entity_value(entity_value=e2.Value):
                    continue

                g1 = self.__synonyms.get_synonym_group_index(e1.Value)
                g2 = self.__synonyms.get_synonym_group_index(e2.Value)
                if g1 == g2:
                    continue

                s_dist = self.__get_distance_in_sentences_between_entities(n_id=news_id, e1=e1, e2=e2)

                if s_dist > 0:
                    continue

                t_dist = TextOpinionHelper.calculate_distance_between_entities_in_terms(
                    parsed_news=self.__pnc.get_by_news_id(news_id),
                    e1=e1,
                    e2=e2)

                if t_dist > 10:
                    continue

                if sentiment_opinions is not None:
                    o = Opinion(source_value=e1.Value,
                                target_value=e2.Value,
                                sentiment=NeutralLabel())
                    if sentiment_opinions.has_synonymous_opinion(opinion=o):
                        continue

                key = self.__create_key_by_entity_pair(e1=e1, e2=e2)
                relevant_pairs[key] = 0

        opinions = self.__create_opinions_between_entities(
            relevant_pairs=relevant_pairs,
            entities_collection=entities_collection)

        return opinions
