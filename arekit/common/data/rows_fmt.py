from arekit.common.data import const
from arekit.common.utils import filter_whitespaces, split_by_whitespaces


def process_values_list(value, args_sep):
    return value.split(args_sep)


def process_indices_list(value, no_value_func, args_sep):
    return no_value_func() if not value else [int(v) for v in str(value).split(args_sep)]


def process_text(value):
    """ The core method of the input text processing.
    """
    assert(isinstance(value, str) or isinstance(value, list))
    return filter_whitespaces([term for term in split_by_whitespaces(value)]
                              if isinstance(value, str) else value)


def create_base_column_value_fmt(no_value_func=lambda: None, args_sep=","):

    self_func = lambda value: value

    return {
        const.ID: {
            "writer": self_func,
            "parser": self_func
        },
        const.DOC_ID: {
            "writer": self_func,
            "parser": self_func,
        },
        const.S_IND: {
            "writer": self_func,
            "parser": lambda value: int(value)
        },
        const.T_IND: {
            "writer": self_func,
            "parser": lambda value: int(value)
        },
        const.SENT_IND: {
            "writer": self_func,
            "parser": lambda value: int(value)
        },
        const.OPINION_ID: {
            "writer": self_func,
            "parser": lambda value: int(value)
        },
        const.OPINION_LINKAGE_ID: {
            "writer": self_func,
            "parser": lambda value: int(value)
        },
        const.ENTITY_VALUES: {
            "writer": lambda entities: args_sep.join([e.DisplayValue.replace(args_sep, '') for e in entities]),
            "parser": lambda value: process_values_list(value, args_sep=args_sep),
        },
        const.ENTITY_TYPES: {
            "writer": lambda entities: args_sep.join([e.Type.replace(args_sep, '') for e in entities]),
            "parser": lambda value: process_values_list(value, args_sep=args_sep)
        },
        const.ENTITIES: {
            "writer": lambda entity_inds: args_sep.join(entity_inds),
            "parser": lambda value: process_indices_list(value, no_value_func=no_value_func, args_sep=args_sep)
        },
        const.TEXT: {
            "writer": self_func,
            "parser": lambda value: process_text(value)
        },
        const.LABEL_UINT: {
            "writer": self_func,
            "parser": lambda value: int(value)
        }
    }


def create_base_column_fmt(fmt_type, args_sep=","):
    assert(isinstance(fmt_type, str))
    d = create_base_column_value_fmt(args_sep=args_sep)
    for k, v in d.items():
        d[k] = v[fmt_type]
    return d
