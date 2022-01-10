from arekit.contrib.experiment_rusentrel.entities.types import EntityFormattersService
from arekit.contrib.experiment_rusentrel.types import ExperimentTypesService
from arekit.contrib.networks.enum_name_types import ModelNamesService, ModelNames
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersionsService, RuSentiFramesVersions
from arekit.processing.lemmatization.mystem import MystemWrapper
from examples.network.args.base import BaseArg
from examples.network.args.default import EMBEDDING_FILEPATH, TERMS_PER_CONTEXT


class UseBalancingArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.balance_samples[0]

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--balance-samples',
                            dest='balance_samples',
                            type=lambda x: (str(x).lower() == 'true'),
                            nargs=1,
                            help='Use balancing for Train type during sample serialization process"')


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


class EntityFormatterTypesArg(BaseArg):

    @staticmethod
    def read_argument(args):
        name = args.entity_fmt[0]
        return EntityFormattersService.get_type_by_name(name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--entity-fmt',
                            dest='entity_fmt',
                            type=str,
                            choices=list(EntityFormattersService.iter_supported_names()),
                            nargs=1,
                            help='Entity formatter type')


class ExperimentTypeArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        exp_name = args.exp_type
        return ExperimentTypesService.get_type_by_name(exp_name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--experiment',
                            dest='exp_type',
                            type=str,
                            choices=list(ExperimentTypesService.iter_supported_names()),
                            default="rsr",
                            nargs=1,
                            help='Experiment type')


class RuSentiFramesVersionArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return RuSentiFramesVersionsService.get_type_by_name(args.frames_version)

    @staticmethod
    def add_argument(parser):

        default_name = RuSentiFramesVersionsService.get_name_by_type(
            RuSentiFramesVersions.V20)

        parser.add_argument('--frames-version',
                            dest='frames_version',
                            type=str,
                            default=default_name,
                            choices=list(RuSentiFramesVersionsService.iter_supported_names()),
                            nargs='?',
                            help='Version of RuSentiFrames collection (Default: {})'.format(default_name))


class LabelsCountArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.labels_count

    @staticmethod
    def add_argument(parser):
        choices = [2, 3]
        parser.add_argument('--labels-count',
                            dest="labels_count",
                            type=int,
                            choices=[2, 3],
                            default=choices[-1],
                            nargs=1,
                            help="Labels count in an output classifier")


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


class ModelNameArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return ModelNamesService.get_type_by_name(args.model_name)

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--model-name',
                            dest='model_name',
                            type=str,
                            choices=list(ModelNamesService.iter_supported_names()),
                            default=ModelNames.PCNN.value,
                            nargs=1,
                            help='Name of a model to be utilized in experiment')