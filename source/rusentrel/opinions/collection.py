# -*- coding: utf-8 -*-
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.labels.base import Label
from arekit.common.synonyms import SynonymsCollection
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.opinions.opinion import RuSentRelOpinion
from arekit.source.rusentrel.opinions.serializer import RuSentRelOpinionCollectionSerializer


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

    def save_to_file(self, filepath):
        RuSentRelOpinionCollectionSerializer.save_to_file(
            collection=self,
            filepath=filepath)
