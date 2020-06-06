from arekit.contrib.bert_samplers.str_entity_fmt import RussianEntitiesFormatter
from arekit.contrib.bert_samplers.str_label_fmt import RussianThreeScaleLabelsFormatter


def default_labels_formatter():
    return RussianThreeScaleLabelsFormatter()


def default_entities_formatter():
    return RussianEntitiesFormatter()
