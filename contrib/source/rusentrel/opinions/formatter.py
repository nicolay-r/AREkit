import io

from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.opinions.formatter import OpinionCollectionsFormatter
from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import create_dir_if_not_exists
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuSentRelOpinionCollectionFormatter(OpinionCollectionsFormatter):

    def __init__(self, synonyms_collection):
        assert(isinstance(synonyms_collection, SynonymsCollection))
        self.__synonyms = synonyms_collection

    # region protected public methods

    def load_from_file(self, filepath, labels_formatter):
        """
        Important: For externaly saved collections (using save_to_file method) and related usage
        """
        assert(isinstance(filepath, unicode))
        assert(isinstance(labels_formatter, StringLabelsFormatter))

        with open(filepath, 'r') as input_file:
            return RuSentRelOpinionCollectionFormatter._load_from_file(input_file=input_file,
                                                                       labels_formatter=labels_formatter,
                                                                       synonyms=self.__synonyms,
                                                                       is_native_synonyms_collection=False)

    def save_to_file(self, collection, filepath, labels_formatter):
        assert(isinstance(collection, OpinionCollection))
        assert(isinstance(filepath, unicode))
        assert(isinstance(labels_formatter, StringLabelsFormatter))

        def __opinion_key(opinion):
            assert(isinstance(opinion, Opinion))
            return opinion.SourceValue + opinion.TargetValue

        sorted_ops = sorted(collection, key=__opinion_key)

        create_dir_if_not_exists(filepath)

        with io.open(filepath, 'w') as f:
            for o in sorted_ops:

                o_str = RuSentRelOpinionCollectionFormatter.__opinion_to_str(
                    opinion=o,
                    labels_formatter=labels_formatter)

                f.write(o_str)
                f.write(u'\n')

    def save_to_archive(self, collections_iter, labels_formatter):
        raise NotImplementedError()

    # endregion

    # region private methods

    @staticmethod
    def _load_from_file(input_file, synonyms, labels_formatter, is_native_synonyms_collection):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(is_native_synonyms_collection, bool))

        opinions = []
        for i, line in enumerate(input_file.readlines()):

            line = line.decode('utf-8')

            if line == '\n':
                continue

            args = line.strip().split(',')
            assert(len(args) >= 3)

            source_value = args[0].strip()
            target_value = args[1].strip()
            sentiment = labels_formatter.str_to_label(args[2].strip())

            o = Opinion(source_value=source_value,
                        target_value=target_value,
                        sentiment=sentiment)
            opinions.append(o)

        if is_native_synonyms_collection:
            return OpinionCollection(opinions=opinions,
                                     synonyms=synonyms,
                                     raise_exception_on_duplicates=True)
        else:
            return OpinionCollection.init_as_custom(opinions=opinions,
                                                    synonyms=synonyms)

    @staticmethod
    def __opinion_to_str(opinion, labels_formatter):
        assert(isinstance(opinion, Opinion))
        assert(isinstance(labels_formatter, StringLabelsFormatter))

        return u"{}, {}, {}, current".format(
            opinion.SourceValue,
            opinion.TargetValue,
            labels_formatter.label_to_str(opinion.Sentiment))

    # endregion
