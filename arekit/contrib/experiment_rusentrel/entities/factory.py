from arekit.contrib.experiment_rusentrel.entities.types import EntityFormatterTypes
from arekit.contrib.utils.entities.formatters.str_rus_cased_fmt import RussianEntitiesCasedFormatter
from arekit.contrib.utils.entities.formatters.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.utils.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.entities.formatters.str_simple_uppercase_fmt import SimpleUppercasedEntityFormatter


def create_entity_formatter(fmt_type, create_russian_pos_tagger_func=None):
    """ Factory method for entity formatters, applicable in bert.
    """
    assert(isinstance(fmt_type, EntityFormatterTypes))
    assert(callable(create_russian_pos_tagger_func) or create_russian_pos_tagger_func is None)

    if fmt_type == EntityFormatterTypes.RussianCased:
        return RussianEntitiesCasedFormatter(create_russian_pos_tagger_func())
    elif fmt_type == EntityFormatterTypes.HiddenBertStyled:
        return SharpPrefixedEntitiesSimpleFormatter()
    elif fmt_type == EntityFormatterTypes.HiddenSimpleRus:
        return RussianEntitiesFormatter()
    elif fmt_type == EntityFormatterTypes.HiddenSimpleEng:
        return StringEntitiesSimpleFormatter()
    elif fmt_type == EntityFormatterTypes.HiddenSimpleUppercase:
        return SimpleUppercasedEntityFormatter()

