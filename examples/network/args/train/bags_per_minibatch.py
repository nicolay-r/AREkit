from examples.network.args.base import BaseArg
from examples.network.args.default import BAGS_PER_MINIBATCH


class BagsPerMinibatchArg(BaseArg):

    default = BAGS_PER_MINIBATCH

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.bags_per_minibatch

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--bags-per-minibatch',
                            dest='bags_per_minibatch',
                            type=int,
                            default=BagsPerMinibatchArg.default,
                            nargs='?',
                            help='Bags per minibatch count (Default: {})'.format(BagsPerMinibatchArg.default))
