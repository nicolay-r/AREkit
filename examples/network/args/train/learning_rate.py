from examples.network.args.base import BaseArg
from examples.network.args.default import LEARNING_RATE


class LearningRateArg(BaseArg):

    default = LEARNING_RATE

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.learning_rate

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--learning-rate',
                            dest='learning_rate',
                            type=float,
                            default=LearningRateArg.default,
                            nargs='?',
                            help='Learning Rate (Default: {})'.format(LearningRateArg.default))
