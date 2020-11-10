from arekit.common.entities.formatters.str_rus_cased_fmt import RussianEntitiesCasedFormatter
from arekit.common.entities.formatters.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.common.entities.formatters.str_simple_fmt import StringSimpleFormatter
from arekit.common.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.common.entities.formatters.str_simple_uppercase_fmt import SimpleUppercasedEntityFormatter
from arekit.common.entities.formatters.types import EntityFormatterTypes


def create_bert_entity_formatter(fmt_type, create_russian_pos_tagger_func=None):
    """ Factory method for entity formatters, applicable in bert.
    """
    assert(isinstance(fmt_type, EntityFormatterTypes))
    assert(callable(create_russian_pos_tagger_func) or create_russian_pos_tagger_func is None)

    if fmt_type == EntityFormatterTypes.RussianCased:
        return RussianEntitiesCasedFormatter(create_russian_pos_tagger_func())
    elif fmt_type == EntityFormatterTypes.SimpleSharpPrefixed:
        return SharpPrefixedEntitiesSimpleFormatter()
    elif fmt_type == EntityFormatterTypes.RussianSimple:
        return RussianEntitiesFormatter()
    elif fmt_type == EntityFormatterTypes.Simple:
        return StringSimpleFormatter()
    elif fmt_type == EntityFormatterTypes.SimpleUppercase:
        return SimpleUppercasedEntityFormatter()

