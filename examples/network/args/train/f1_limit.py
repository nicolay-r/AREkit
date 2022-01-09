from examples.network.args.base import BaseArg
from examples.network.args.default import TRAIN_F1_LIMIT


class TrainF1LimitArg(BaseArg):

    default = TRAIN_F1_LIMIT

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.train_f1_limit

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--train-f1-limit',
                            dest='train_f1_limit',
                            type=float,
                            default=TrainF1LimitArg.default,
                            nargs='?',
                            help="Train Accuracy Limit (Default: {})".format(TrainF1LimitArg.default))
