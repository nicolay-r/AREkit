from examples.network.args.base import BaseArg
from examples.network.args.default import TEST_EVERY_K_EPOCH


class TestEveryEpochsCountArg(BaseArg):

    default = TEST_EVERY_K_EPOCH

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.test_every

    @staticmethod
    def add_argument(parser):
        parser.add_argument(
            '--test-every',
            dest='test_every',
            type=int,
            default=TestEveryEpochsCountArg.default,
            nargs='?',
            help="Test every i'th epoch (Default: every {}'th epoch)".format(TestEveryEpochsCountArg.default))
