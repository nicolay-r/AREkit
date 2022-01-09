from examples.network.args.base import BaseArg


class LabelsCountArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.labels_count[0]

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--labels-count',
                            dest="labels_count",
                            type=int,
                            choices=[2, 3],
                            nargs=1,
                            help="Labels count in an output classifier")
