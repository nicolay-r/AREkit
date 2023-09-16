from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.opinions.base import Opinion


class OpinionConverter(object):
    """ Opinion type <-> string Converter.
    """

    @staticmethod
    def try_from_string(line, labels_formatter):
        assert(isinstance(line, str))

        args = line.strip().split(',')
        assert (len(args) >= 3)

        source_value = args[0].strip()
        target_value = args[1].strip()
        str_label = args[2].strip()

        if not labels_formatter.supports_value(str_label):
            return None

        return Opinion(source_value=source_value,
                       target_value=target_value,
                       label=labels_formatter.str_to_label(str_label))

    @staticmethod
    def try_to_string(opinion, labels_formatter):
        assert(isinstance(opinion, Opinion))
        assert(isinstance(labels_formatter, StringLabelsFormatter))

        label = opinion.Label

        if not labels_formatter.supports_label(label):
            return None

        return "{}, {}, {}, current".format(
            opinion.SourceValue,
            opinion.TargetValue,
            labels_formatter.label_to_str(opinion.Label))
