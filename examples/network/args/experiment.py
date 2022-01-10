from arekit.contrib.experiment_rusentrel.types import ExperimentTypesService, ExperimentTypes
from examples.network.args.base import BaseArg


class ExperimentTypeArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        exp_name = args.exp_type
        return ExperimentTypesService.get_type_by_name(exp_name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--experiment',
                            dest='exp_type',
                            type=str,
                            choices=list(ExperimentTypesService.iter_supported_names()),
                            default="rsr",
                            nargs=1,
                            help='Experiment type')
