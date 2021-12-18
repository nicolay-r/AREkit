from arekit.contrib.experiment_rusentrel.entities.str_rus_cased_fmt import RussianEntitiesCasedFormatter
from arekit.contrib.experiment_rusentrel.entities.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.experiment_rusentrel.entities.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.experiment_rusentrel.entities.str_simple_sharp_prefixed_fmt import \
    SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.experiment_rusentrel.entities.str_simple_uppercase_fmt import SimpleUppercasedEntityFormatter
from arekit.contrib.experiment_rusentrel.entities.types import EntityFormatterTypes


def create_entity_formatter(fmt_type, create_russian_pos_tagger_func=None):
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
        return StringEntitiesSimpleFormatter()
    elif fmt_type == EntityFormatterTypes.SimpleUppercase:
        return SimpleUppercasedEntityFormatter()

