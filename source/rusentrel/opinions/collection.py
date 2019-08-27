# -*- coding: utf-8 -*-
import io
from core.common.opinions.collection import OpinionCollection
from core.common.opinions.opinion import Opinion
from core.common.utils import create_dir_if_not_exists
from core.evaluation.labels import Label
from core.common.synonyms import SynonymsCollection
from core.source.rusentrel.io_utils import RuSentRelIOUtils
from core.source.rusentrel.opinions.opinion import RuSentRelOpinion


class RuSentRelOpinionCollection(OpinionCollection):
    """
    Collection of sentiment opinions between entities
    """

    @classmethod
    def read_collection(cls, doc_id, synonyms):
        return RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_sentiment_opin_filepath(doc_id),
            process_func=lambda input_file: cls.__from_file(input_file, synonyms))

    @classmethod
    def read_from_file(cls, filepath, synonyms):
        """
        Important: For externaly saved collections (using save_to_file method) and related usage
        """
        assert(isinstance(filepath, unicode))
        assert(isinstance(synonyms, SynonymsCollection))

        with open(filepath, 'r') as input_file:
            return cls.__from_file(input_file, synonyms=synonyms)

    @classmethod
    def __from_file(cls, input_file, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))

        opinions = []
        for i, line in enumerate(input_file.readlines()):

            line = line.decode('utf-8')

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

    @staticmethod
    def __opinion_to_str(opinion):
        assert(isinstance(opinion, RuSentRelOpinion))
        return u"{}, {}, {}, current".format(
            opinion.SourceValue,
            opinion.TargetValue,
            opinion.Sentiment.to_str())

    def save_to_file(self, filepath):
        assert(isinstance(filepath, unicode))

        def __opinion_key(opinion):
            assert(isinstance(opinion, Opinion))
            return opinion.SourceValue + opinion.TargetValue

        sorted_ops = sorted(self, key=__opinion_key)

        create_dir_if_not_exists(filepath)

        with io.open(filepath, 'w') as f:
            for o in sorted_ops:
                f.write(self.__opinion_to_str(o))
                f.write(u'\n')
