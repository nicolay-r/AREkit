from examples.network.args.base import BaseArg


class RusVectoresEmbeddingFilepathArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.embedding_filepath[0]

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--emb-filepath',
                            dest='embedding_filepath',
                            type=str,
                            nargs=1,
                            help='RusVectores embedding filepath')
