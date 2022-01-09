from arekit.contrib.networks.enum_input_types import ModelInputType, ModelInputTypeService
from examples.network.args.base import BaseArg


class ModelInputTypeArg(BaseArg):

    _default = ModelInputType.SingleInstance

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return ModelInputTypeService.get_type_by_name(args.input_type)

    @staticmethod
    def add_argument(parser):
        str_def = ModelInputTypeService.find_name_by_type(ModelInputTypeArg._default)
        parser.add_argument('--model-input-type',
                            dest='input_type',
                            type=str,
                            choices=list(ModelInputTypeService.iter_supported_names()),
                            default=str_def,
                            nargs='?',
                            help='Input format type (Default: {})'.format(str_def))
