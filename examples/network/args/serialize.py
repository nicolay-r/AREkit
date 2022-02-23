from arekit.contrib.experiment_rusentrel.entities.types import EntityFormattersService
from examples.network.args.base import BaseArg


class EntityFormatterTypesArg(BaseArg):

    @staticmethod
    def read_argument(args):
        name = args.entity_fmt
        return EntityFormattersService.get_type_by_name(name)

    @staticmethod
    def add_argument(parser, default):
        assert(EntityFormattersService.is_supported(default))
        parser.add_argument('--entity-fmt',
                            dest='entity_fmt',
                            type=str,
                            choices=list(EntityFormattersService.iter_supported_names()),
                            default='simple',
                            nargs=1,
                            help='Entity formatter type')
