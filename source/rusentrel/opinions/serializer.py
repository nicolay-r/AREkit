import io

from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.opinions.serializer import OpinionCollectionSerializer
from arekit.common.utils import create_dir_if_not_exists


class RuSentRelOpinionCollectionSerializer(OpinionCollectionSerializer):

    @staticmethod
    def __opinion_to_str(opinion):
        assert(isinstance(opinion, Opinion))
        return u"{}, {}, {}, current".format(
            opinion.SourceValue,
            opinion.TargetValue,
            opinion.Sentiment.to_str())

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
                f.write(RuSentRelOpinionCollectionSerializer.__opinion_to_str(o))
                f.write(u'\n')
