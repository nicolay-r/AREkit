from arekit.processing.lemmatization.mystem import MystemWrapper
from examples.network.args.base import BaseArg


class StemmerArg(BaseArg):

    default = u"mystem"

    supported = {
        u"mystem": MystemWrapper()
    }

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return StemmerArg.supported[args.stemmer]

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--stemmer',
                            dest='stemmer',
                            type=str,
                            choices=list(StemmerArg.supported.keys()),
                            default=StemmerArg.default,
                            nargs='?',
                            help='Stemmer (Default: {})'.format(StemmerArg.default))
