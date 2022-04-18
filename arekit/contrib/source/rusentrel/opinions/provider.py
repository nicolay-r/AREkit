from arekit.common.opinions.provider import OpinionCollectionsProvider
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.rusentrel.opinions.converter import OpinionConverter


class RuSentRelOpinionCollectionProvider(OpinionCollectionsProvider):

    @staticmethod
    def _iter_opinions_from_file(input_file, labels_formatter, error_on_non_supported):
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(error_on_non_supported, bool))

        for line in input_file.readlines():

            # Force perform decoding if needed.
            if isinstance(line, bytes):
                line = line.decode()

            if line == '\n':
                continue

            str_opinion = OpinionConverter.try_from_string(
                line=line,
                labels_formatter=labels_formatter)

            if str_opinion is None:
                if error_on_non_supported:
                    raise Exception("Line '{line}' has non supported label")
                else:
                    continue

            yield str_opinion

    # region public methods

    def iter_opinions(self, source, encoding, labels_formatter, error_on_non_supported=True):
        """
        Important: For externally saved collections (using save_to_file method) and related usage
        """
        assert(isinstance(source, str))
        assert(isinstance(labels_formatter, StringLabelsFormatter))
        assert(isinstance(error_on_non_supported, bool))

        with open(source, 'r', encoding=encoding) as input_file:

            it = RuSentRelOpinionCollectionProvider._iter_opinions_from_file(
                input_file=input_file,
                labels_formatter=labels_formatter,
                error_on_non_supported=error_on_non_supported)

            for opinion in it:
                yield opinion

    # endregion
