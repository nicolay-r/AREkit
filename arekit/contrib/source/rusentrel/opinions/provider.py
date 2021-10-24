from arekit.common.opinions.base import Opinion
from arekit.common.opinions.provider import OpinionCollectionsProvider
from arekit.common.labels.str_fmt import StringLabelsFormatter


class RuSentRelOpinionCollectionProvider(OpinionCollectionsProvider):

    # region private methods

    @staticmethod
    def __try_str_to_opinion(line, labels_formatter):
        args = line.strip().split(',')
        assert (len(args) >= 3)

        source_value = args[0].strip()
        target_value = args[1].strip()
        str_label = args[2].strip()

        if not labels_formatter.supports_value(str_label):
            return None

        return Opinion(source_value=source_value,
                       target_value=target_value,
                       sentiment=labels_formatter.str_to_label(str_label))

    # endregion

    @staticmethod
    def _iter_opinions_from_file(input_file, labels_formatter, error_on_non_supported):
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(error_on_non_supported, bool))

        for line in input_file.readlines():

            line = line.decode('utf-8')

            if line == '\n':
                continue

            str_opinion = RuSentRelOpinionCollectionProvider.__try_str_to_opinion(
                line=line,
                labels_formatter=labels_formatter)

            if str_opinion is None:
                if error_on_non_supported:
                    raise Exception("Line '{line}' has non supported label")
                else:
                    continue

            yield str_opinion

    # region public methods

    def iter_opinions(self, source, labels_formatter, error_on_non_supported=True):
        """
        Important: For externaly saved collections (using save_to_file method) and related usage
        """
        assert(isinstance(source, str))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(error_on_non_supported, bool))

        with open(source, 'r') as input_file:

            it = RuSentRelOpinionCollectionProvider._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=labels_formatter,
                error_on_non_supported=error_on_non_supported)

            for opinion in it:
                yield opinion

    # endregion
