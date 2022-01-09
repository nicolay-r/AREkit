from examples.network.args.base import BaseArg
from examples.network.args.default import TRAIN_ACC_LIMIT


class TrainAccuracyLimitArg(BaseArg):

    default = TRAIN_ACC_LIMIT

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.train_acc_limit

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--train-acc-limit',
                            dest='train_acc_limit',
                            type=float,
                            default=TrainAccuracyLimitArg.default,
                            nargs='?',
                            help="Train Accuracy Limit (Default: {})".format(TrainAccuracyLimitArg.default))
