# -*- coding: utf-8 -*-
import io
from core.common.opinions.collection import OpinionCollection
from core.evaluation.labels import Label
from core.common.opinions.opinion import Opinion
from core.common.synonyms import SynonymsCollection


class RuSentRelOpinionCollection(OpinionCollection):
    """ Collection of sentiment opinions between entities
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

                entity_left = args[0].strip()
                entity_right = args[1].strip()
                sentiment = Label.from_str(args[2].strip())

                o = Opinion(entity_left, entity_right, sentiment)
                opinions.append(o)

        return cls(opinions, synonyms)

    def save(self, filepath):
        assert(isinstance(filepath, unicode))

        sorted_ops = sorted(self.__opinions,
                            key=lambda o: o.value_left + o.value_right)

        with io.open(filepath, 'w') as f:
            for o in sorted_ops:
                f.write(self.__opinion_to_str(o))
                f.write(u'\n')

    @staticmethod
    def __opinion_to_str(opinion):
        assert(isinstance(opinion, Opinion))
        return u"{}, {}, {}, current".format(
            opinion.ValueLeft,
            opinion.ValueRight,
            opinion.Sentiment.to_str())
