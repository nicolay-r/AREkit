from examples.network.args.base import BaseArg


class ModelNameTagArg(BaseArg):

    NO_TAG = u''

    default = NO_TAG

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return str(args.model_tag)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--model-tag',
                            dest='model_tag',
                            type=str,
                            default=ModelNameTagArg.NO_TAG,
                            nargs='?',
                            help='Optional and additional custom model name suffix.')
