from examples.network.args.base import BaseArg


class DistanceInTermsBetweenAttitudeEndsArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.dist_between_ends

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--dist-between-att-ends',
                            dest='dist_between_ends',
                            type=int,
                            default=None,
                            nargs='?',
                            help='Distance in terms between attitude participants in terms.'
                                 '(Default: {})'.format(None))
