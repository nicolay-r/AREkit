# -*- coding: utf-8 -*-
import io
from core.common.opinions.collection import OpinionCollection
from core.evaluation.labels import Label
from core.common.synonyms import SynonymsCollection
from core.source.rusentrel.opinions.opinion import RuSentRelOpinion


class RuSentRelOpinionCollection(OpinionCollection):
    """
    Collection of sentiment opinions between entities
    """

    @classmethod
    def from_file(cls, filepath, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))

        opinions = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):

                if line == '\n':
                    continue

                args = line.strip().split(',')
                assert(len(args) >= 3)

                value_source = args[0].strip()
                value_target = args[1].strip()
                sentiment = Label.from_str(args[2].strip())

                o = RuSentRelOpinion(value_source=value_source,
                                     value_target=value_target,
                                     sentiment=sentiment)
                opinions.append(o)

        return cls(opinions, synonyms)

    def save(self, filepath):
        assert(isinstance(filepath, unicode))

        def __opinion_key(opinion):
            assert(isinstance(opinion, RuSentRelOpinion))
            return opinion.SourceValue + opinion.TargetValue

        sorted_ops = sorted(self, key=__opinion_key)

        with io.open(filepath, 'w') as f:
            for o in sorted_ops:
                f.write(self.__opinion_to_str(o))
                f.write(u'\n')

    @staticmethod
    def __opinion_to_str(opinion):
        assert(isinstance(opinion, RuSentRelOpinion))
        return u"{}, {}, {}, current".format(
            opinion.SourceValue,
            opinion.TargetValue,
            opinion.Sentiment.to_str())
