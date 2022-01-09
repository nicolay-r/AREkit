from arekit.contrib.experiment_rusentrel.entities.types import EntityFormattersService
from examples.network.args.base import BaseArg


class EnitityFormatterTypesArg(BaseArg):

    @staticmethod
    def read_argument(args):
        name = args.entity_fmt[0]
        return EntityFormattersService.get_type_by_name(name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--entity-fmt',
                            dest='entity_fmt',
                            type=str,
                            choices=list(EntityFormattersService.iter_supported_names()),
                            nargs=1,
                            help='Entity formatter type')
