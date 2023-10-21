import arekit.contrib.networks.input.const as const
from arekit.common.data.rows_fmt import process_indices_list


def create_nn_column_formatters(no_value_func=lambda: None, args_sep=","):
    assert(callable(no_value_func))

    empty_list = []

    def str_to_list(value):
        return process_indices_list(value, no_value_func=no_value_func, args_sep=args_sep)

    def list_to_str(inds_iter):
        return args_sep.join([str(i) for i in inds_iter])

    return {
        const.FrameVariantIndices: {
            "writer": lambda value: list_to_str(value),
            "parser": lambda value: process_indices_list(value, no_value_func=no_value_func, args_sep=args_sep)
                if isinstance(value, str) else empty_list
        },
        const.FrameConnotations: {
            "writer": lambda value: list_to_str(value),
            "parser": lambda value: process_indices_list(value, no_value_func=no_value_func, args_sep=args_sep)
                if isinstance(value, str) else empty_list
        },
        const.SynonymObject: {
            "writer": lambda value: list_to_str(value),
            "parser": lambda value: process_indices_list(value, no_value_func=no_value_func, args_sep=args_sep)
        },
        const.SynonymSubject: {
            "writer": lambda value: list_to_str(value),
            "parser": lambda value: process_indices_list(value, no_value_func=no_value_func, args_sep=args_sep)
        },
        const.PosTags: {
            "writer": lambda value: list_to_str(value),
            "parser": lambda value: str_to_list(value)
        }
    }


def create_nn_val_writer_fmt(fmt_type, args_sep=","):
    assert(isinstance(fmt_type, str))
    d = create_nn_column_formatters(args_sep=args_sep)
    for k, v in d.items():
        d[k] = v[fmt_type]
    return d
