import io

from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.opinions.writer import OpinionCollectionWriter
from arekit.common.utils import create_dir_if_not_exists


class RuSentRelOpinionCollectionWriter(OpinionCollectionWriter):

    @staticmethod
    def __try_opinion_to_str(opinion, labels_formatter):
        assert(isinstance(opinion, Opinion))
        assert(isinstance(labels_formatter, StringLabelsFormatter))

        label = opinion.Sentiment

        if not labels_formatter.supports_label(label):
            return None

        return "{}, {}, {}, current".format(
            opinion.SourceValue,
            opinion.TargetValue,
            labels_formatter.label_to_str(opinion.Sentiment))

    def serialize(self, collection, target, labels_formatter, error_on_non_supported=True):
        assert(isinstance(collection, OpinionCollection))
        assert(isinstance(target, str))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(error_on_non_supported, bool))

        def __opinion_key(opinion):
            assert (isinstance(opinion, Opinion))
            return opinion.SourceValue + opinion.TargetValue

        sorted_ops = sorted(collection, key=__opinion_key)

        create_dir_if_not_exists(target)

        with io.open(target, 'w') as f:
            for o in sorted_ops:

                str_value = RuSentRelOpinionCollectionWriter.__try_opinion_to_str(
                    opinion=o,
                    labels_formatter=labels_formatter)

                if str_value is None:
                    if error_on_non_supported:
                        raise Exception("Opinion label `{label}` is not supported by formatter".format(
                            label=o.Sentiment))
                    else:
                        continue

                f.write(str_value)
                f.write('\n')
