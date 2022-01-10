from arekit.contrib.networks.enum_input_types import ModelInputType, ModelInputTypeService
from examples.network.args.base import BaseArg
from examples.network.args import const


class TrainAccuracyLimitArg(BaseArg):

    default = const.TRAIN_ACC_LIMIT

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.train_acc_limit

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--train-acc-limit',
                            dest='train_acc_limit',
                            type=float,
                            default=TrainAccuracyLimitArg.default,
                            nargs='?',
                            help="Train Accuracy Limit (Default: {})".format(TrainAccuracyLimitArg.default))


class BagsPerMinibatchArg(BaseArg):

    default = const.BAGS_PER_MINIBATCH

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


class DropoutKeepProbArg(BaseArg):

    default = const.DROPOUT_KEEP_PROB

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.dropout_keep_prob

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--dropout-keep-prob',
                            dest='dropout_keep_prob',
                            type=float,
                            default=DropoutKeepProbArg.default,
                            nargs='?',
                            help='Dropout keep prob (Default: {})'.format(DropoutKeepProbArg.default))


class EpochsCountArg(BaseArg):

    default = const.EPOCHS_COUNT

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.epochs

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--epochs',
                            dest='epochs',
                            type=int,
                            default=EpochsCountArg.default,
                            nargs='?',
                            help='Epochs count (Default: {})'.format(EpochsCountArg.default))


class TrainF1LimitArg(BaseArg):

    default = const.TRAIN_F1_LIMIT

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.train_f1_limit

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--train-f1-limit',
                            dest='train_f1_limit',
                            type=float,
                            default=TrainF1LimitArg.default,
                            nargs='?',
                            help="Train Accuracy Limit (Default: {})".format(TrainF1LimitArg.default))


class LearningRateArg(BaseArg):

    default = const.LEARNING_RATE

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.learning_rate

    @staticmethod
    def add_argument(parser):
        parser.add_argument('--learning-rate',
                            dest='learning_rate',
                            type=float,
                            default=LearningRateArg.default,
                            nargs='?',
                            help='Learning Rate (Default: {})'.format(LearningRateArg.default))


class ModelInputTypeArg(BaseArg):

    _default = ModelInputType.SingleInstance

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return ModelInputTypeService.get_type_by_name(args.input_type)

    @staticmethod
    def add_argument(parser):
        str_def = ModelInputTypeService.find_name_by_type(ModelInputTypeArg._default)
        parser.add_argument('--model-input-type',
                            dest='input_type',
                            type=str,
                            choices=list(ModelInputTypeService.iter_supported_names()),
                            default=str_def,
                            nargs='?',
                            help='Input format type (Default: {})'.format(str_def))


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
