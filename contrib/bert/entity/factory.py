from arekit.contrib.bert.entity.str_rus_cased_fmt import RussianEntitiesCasedFormatter
from arekit.contrib.bert.entity.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.bert.entity.str_simple_fmt import EntitiesSimpleFormatter
from arekit.contrib.bert.entity.types import BertEntityFormatterTypes


def create_bert_entity_formatter(fmt_type, create_russian_pos_tagger_func=None):
    """ Factory method for entity formatters, applicable in bert.
    """
    assert(isinstance(fmt_type, BertEntityFormatterTypes))
    assert(callable(create_russian_pos_tagger_func) or create_russian_pos_tagger_func is None)

    if fmt_type == BertEntityFormatterTypes.RussianCased:
        return RussianEntitiesCasedFormatter(create_russian_pos_tagger_func())
    elif fmt_type == BertEntityFormatterTypes.RussianSimple:
        return EntitiesSimpleFormatter()
    elif fmt_type == BertEntityFormatterTypes.Simple:
        return RussianEntitiesFormatter()

