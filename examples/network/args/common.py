from arekit.contrib.experiment_rusentrel.types import ExperimentTypesService
from arekit.contrib.networks.enum_name_types import ModelNamesService, ModelNames
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersionsService, RuSentiFramesVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from examples.network.args.base import BaseArg
from examples.network.args import const


class InputTextArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.input_text

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, str))
        parser.add_argument('--text',
                            dest='input_text',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Input text for processing')


class PredictOutputFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.inference_output_filepath

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('-o',
                            dest='inference_output_filepath',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Inference output filepath')


class VocabFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.vocab_filepath

    @staticmethod
    def add_argument(parser, default=None):
        parser.add_argument('--vocab-filepath',
                            dest='vocab_filepath',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Custom vocabulary filepath')


class UseBalancingArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.balance_samples

    @staticmethod
    def add_argument(parser, default="True"):
        parser.add_argument('--balance-samples',
                            dest='balance_samples',
                            type=lambda x: (str(x).lower() == 'true'),
                            default=default,
                            nargs=1,
                            help='Use balancing for Train type during sample serialization process"')


class DistanceInTermsBetweenAttitudeEndsArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.dist_between_ends

    @staticmethod
    def add_argument(parser, default=None):
        parser.add_argument('--dist-between-att-ends',
                            dest='dist_between_ends',
                            type=int,
                            default=default,
                            nargs='?',
                            help='Distance in terms between attitude participants in terms.'
                                 '(Default: {})'.format(None))


class RusVectoresEmbeddingFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.embedding_filepath

    @staticmethod
    def add_argument(parser, default=const.EMBEDDING_FILEPATH):
        parser.add_argument('--emb-filepath',
                            dest='embedding_filepath',
                            type=str,
                            default=default,
                            nargs=1,
                            help='RusVectores embedding filepath')


class EmbeddingMatrixFilepathArg(BaseArg):
    """ Embedding matrix, utilized as an input for model.
    """

    @staticmethod
    def read_argument(args):
        return args.embedding_matrix_filepath

    @staticmethod
    def add_argument(parser, default=None):
        parser.add_argument('--emb-npz-filepath',
                            dest='embedding_matrix_filepath',
                            type=str,
                            default=default,
                            nargs=1,
                            help='RusVectores embedding filepath')


class ExperimentTypeArg(BaseArg):

    @staticmethod
    def read_argument(args):
        exp_name = args.exp_type
        return ExperimentTypesService.get_type_by_name(exp_name)

    @staticmethod
    def add_argument(parser, default="rsr"):
        parser.add_argument('--experiment',
                            dest='exp_type',
                            type=str,
                            choices=list(ExperimentTypesService.iter_supported_names()),
                            default=default,
                            nargs=1,
                            help='Experiment type')


class RuSentiFramesVersionArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return RuSentiFramesVersionsService.get_type_by_name(args.frames_version)

    @staticmethod
    def add_argument(parser, default=RuSentiFramesVersionsService.get_name_by_type(RuSentiFramesVersions.V20)):

        parser.add_argument('--frames-version',
                            dest='frames_version',
                            type=str,
                            default=default,
                            choices=list(RuSentiFramesVersionsService.iter_supported_names()),
                            nargs='?',
                            help='Version of RuSentiFrames collection (Default: {})'.format(default))


class LabelsCountArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.labels_count

    @staticmethod
    def add_argument(parser, default=3):
        choices = [2, default]
        parser.add_argument('--labels-count',
                            dest="labels_count",
                            type=int,
                            choices=choices,
                            default=default,
                            nargs=1,
                            help="Labels count in an output classifier")


class StemmerArg(BaseArg):

    supported = {
        u"mystem": MystemWrapper()
    }

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return StemmerArg.supported[args.stemmer]

    @staticmethod
    def add_argument(parser, default=u"mystem"):
        assert(default in StemmerArg.supported)
        parser.add_argument('--stemmer',
                            dest='stemmer',
                            type=str,
                            choices=list(StemmerArg.supported.keys()),
                            default=default,
                            nargs='?',
                            help='Stemmer (Default: {})'.format(default))


class TermsPerContextArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.terms_per_context

    @staticmethod
    def add_argument(parser, default=const.TERMS_PER_CONTEXT):
        parser.add_argument('--terms-per-context',
                            dest='terms_per_context',
                            type=int,
                            default=default,
                            nargs='?',
                            help='The max possible length of an input context in terms (Default: {})\n'
                                 'NOTE: Use greater or equal value for this parameter during experiment'
                                 'process; otherwise you may encounter with exception during sample '
                                 'creation process!'.format(default))


class ModelNameArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return ModelNamesService.get_type_by_name(args.model_name)

    @staticmethod
    def add_argument(parser, default=ModelNames.PCNN.value):
        parser.add_argument('--model-name',
                            dest='model_name',
                            type=str,
                            choices=list(ModelNamesService.iter_supported_names()),
                            default=default,
                            nargs=1,
                            help='Name of a model to be utilized in experiment')


class ModelLoadDirArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.model_load_dir

    @staticmethod
    def add_argument(parser, default=None):
        parser.add_argument('--model-state-dir',
                            dest='model_load_dir',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Use pretrained state as initial')