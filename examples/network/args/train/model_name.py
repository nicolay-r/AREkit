from arekit.contrib.networks.enum_name_types import ModelNamesService, ModelNames
from examples.network.args.base import BaseArg


class ModelNameArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return ModelNamesService.get_type_by_name(args.model_name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--model-name',
                            dest='model_name',
                            type=str,
                            choices=list(ModelNamesService.iter_supported_names()),
                            default=ModelNames.PCNN.value,
                            nargs=1,
                            help='Name of a model to be utilized in experiment')
