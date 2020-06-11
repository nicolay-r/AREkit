from arekit.contrib.bert_samplers.entity.str_entity_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.bert_samplers.str_label_fmt import RussianThreeScaleLabelsFormatter


def default_labels_formatter():
    return RussianThreeScaleLabelsFormatter()


def default_entities_formatter():
    return RussianEntitiesFormatter()
