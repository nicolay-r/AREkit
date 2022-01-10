from examples.network.args.base import BaseArg
from examples.network.args.default import EMBEDDING_FILEPATH


class RusVectoresEmbeddingFilepathArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.embedding_filepath

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--emb-filepath',
                            dest='embedding_filepath',
                            type=str,
                            default=EMBEDDING_FILEPATH,
                            nargs=1,
                            help='RusVectores embedding filepath')
