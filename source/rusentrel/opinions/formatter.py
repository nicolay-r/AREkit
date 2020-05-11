import io

from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.opinions.formatter import OpinionCollectionFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import create_dir_if_not_exists


class RuSentRelOpinionCollectionFormatter(OpinionCollectionFormatter):

    @staticmethod
    def load_from_file(filepath, synonyms):
        """
        Important: For externaly saved collections (using save_to_file method) and related usage
        """
        assert(isinstance(filepath, unicode))
        assert(isinstance(synonyms, SynonymsCollection))

        with open(filepath, 'r') as input_file:
            return RuSentRelOpinionCollectionFormatter._load_from_file(input_file, synonyms=synonyms)

    @staticmethod
    def save_to_file(collection, filepath):
        assert(isinstance(collection, OpinionCollection))
        assert(isinstance(filepath, unicode))

        def __opinion_key(opinion):
            assert(isinstance(opinion, Opinion))
            return opinion.SourceValue + opinion.TargetValue

        sorted_ops = sorted(collection, key=__opinion_key)

        create_dir_if_not_exists(filepath)

        with io.open(filepath, 'w') as f:
            for o in sorted_ops:
                f.write(RuSentRelOpinionCollectionFormatter.__opinion_to_str(o))
                f.write(u'\n')

    @staticmethod
    def _load_from_file(input_file, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))

        opinions = []
        for i, line in enumerate(input_file.readlines()):

            line = line.decode('utf-8')

            if line == '\n':
                continue

            args = line.strip().split(',')
            assert(len(args) >= 3)

            source_value = args[0].strip()
            target_value = args[1].strip()
            # TODO. Refactor. (Provide label scaler)
            sentiment = Label.from_str(args[2].strip())

            o = Opinion(source_value=source_value,
                        target_value=target_value,
                        sentiment=sentiment)
            opinions.append(o)

        return OpinionCollection(opinions, synonyms)

    @staticmethod
    def __opinion_to_str(opinion):
        assert(isinstance(opinion, Opinion))
        return u"{}, {}, {}, current".format(
            opinion.SourceValue,
            opinion.TargetValue,
            opinion.Sentiment.to_str())
