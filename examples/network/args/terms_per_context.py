from examples.network.args.base import BaseArg
from examples.network.args.default import TERMS_PER_CONTEXT


class TermsPerContextArg(BaseArg):

    default = TERMS_PER_CONTEXT

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.terms_per_context

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--terms-per-context',
                            dest='terms_per_context',
                            type=int,
                            default=TermsPerContextArg.default,
                            nargs='?',
                            help='The max possible length of an input context in terms (Default: {})\n'
                                 'NOTE: Use greater or equal value for this parameter during experiment'
                                 'process; otherwise you may encounter with exception during sample '
                                 'creation process!'.format(TermsPerContextArg.default))
