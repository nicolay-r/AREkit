import argparse

from arekit.common.experiment.api.ctx_training import TrainingData
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.factory import create_experiment
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.run_training import NetworksTrainingEngine
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService

from examples.input import EXAMPLES

from examples.network.args.default import BAG_SIZE
from examples.network.args.dist_in_terms_between_ends import DistanceInTermsBetweenAttitudeEndsArg
from examples.network.args.experiment import ExperimentTypeArg
from examples.network.args.labels_count import LabelsCountArg
from examples.network.args.stemmer import StemmerArg
from examples.network.args.terms_per_context import TermsPerContextArg
from examples.network.args.train.acc_limit import TrainAccuracyLimitArg
from examples.network.args.train.bags_per_minibatch import BagsPerMinibatchArg
from examples.network.args.train.dropout_keep_prob import DropoutKeepProbArg
from examples.network.args.train.epochs_count import EpochsCountArg
from examples.network.args.train.f1_limit import TrainF1LimitArg
from examples.network.args.train.learning_rate import LearningRateArg
from examples.network.args.train.model_input_type import ModelInputTypeArg
from examples.network.args.train.model_name import ModelNameArg
from examples.network.args.train.model_name_tag import ModelNameTagArg
from examples.network.common import Common
from examples.network.factory_bags_collection import create_bags_collection_type
from examples.network.factory_config_setups import optionally_modify_config_for_experiment, modify_config_for_model
from examples.network.factory_networks import compose_network_and_network_config_funcs
from examples.network.utils import CustomIOUtils

if __name__ == '__main__':

    text = EXAMPLES["simple"]

    parser = argparse.ArgumentParser(description="Training script for obtaining Tensorflow based states")

    # Composing cmd arguments.
    LabelsCountArg.add_argument(parser)
    ExperimentTypeArg.add_argument(parser)
    StemmerArg.add_argument(parser)
    DropoutKeepProbArg.add_argument(parser)
    BagsPerMinibatchArg.add_argument(parser)
    TermsPerContextArg.add_argument(parser)
    LearningRateArg.add_argument(parser)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser)
    TrainAccuracyLimitArg.add_argument(parser)
    TrainF1LimitArg.add_argument(parser)
    ModelInputTypeArg.add_argument(parser)
    ModelNameArg.add_argument(parser)
    ModelNameTagArg.add_argument(parser)
    EpochsCountArg.add_argument(parser)

    parser.add_argument('--model-state-dir',
                        dest='model_load_dir',
                        type=str,
                        default=None,
                        nargs='?',
                        help='Use pretrained state as initial')

    parser.add_argument('--emb-filepath',
                        dest='embedding_filepath',
                        type=str,
                        nargs='?',
                        help='Custom embedding filepath')

    parser.add_argument('--vocab-filepath',
                        dest='vocab_filepath',
                        type=str,
                        nargs='?',
                        help='Custom vocabulary filepath')

    parser.add_argument('--balanced-input',
                        dest='balanced_input',
                        type=lambda x: (str(x).lower() == 'true'),
                        nargs=1,
                        help='Balanced input of the Train set"')

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    model_input_type = ModelInputTypeArg.read_argument(args)
    model_load_dir = args.model_load_dir
    model_name = ModelNameArg.read_argument(args)
    embedding_filepath = args.embedding_filepath
    vocab_filepath = args.vocab_filepath
    dropout_keep_prob = DropoutKeepProbArg.read_argument(args)
    bags_per_minibatch = BagsPerMinibatchArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    learning_rate = LearningRateArg.read_argument(args)
    test_every_k_epoch = args.test_every_k_epoch
    balanced_input = args.balanced_input[0]
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    train_acc_limit = TrainAccuracyLimitArg.read_argument(args)
    train_f1_limit = TrainF1LimitArg.read_argument(args)
    model_name_tag = ModelNameTagArg.read_argument(args)
    epochs_count = EpochsCountArg.read_argument(args)

    # Utilize predefined versions and folding format.
    rusentrel_version = RuSentRelVersions.V11
    ra_version = RuAttitudesVersions.V20LargeNeut
    folding_type = FoldingType.Fixed

    # init handler
    bags_collection_type = create_bags_collection_type(model_input_type=model_input_type)
    network_func, network_config_func = compose_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=model_input_type)

    labels_scaler = Common.create_labels_scaler(labels_count)

    # Creating experiment
    experiment_data = TrainingData(labels_count=labels_scaler.LabelsCount)

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=balanced_input,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    experiment = create_experiment(exp_type=exp_type,
                                   experiment_data=experiment_data,
                                   folding_type=folding_type,
                                   rusentrel_version=rusentrel_version,
                                   ruattitudes_version=ra_version,
                                   experiment_io_type=CustomIOUtils,
                                   extra_name_suffix=extra_name_suffix,
                                   load_ruattitude_docs=False)

    full_model_name = Common.create_full_model_name(folding_type=folding_type,
                                                    model_name=model_name,
                                                    input_type=model_input_type)

    model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                    target_dir=experiment.ExperimentIO.get_target_dir(),
                                    source_dir=model_load_dir,
                                    embedding_filepath=embedding_filepath,
                                    vocab_filepath=vocab_filepath,
                                    model_name_tag=model_name_tag)

    # Setup model io.
    experiment_data.set_model_io(model_io)

    ###################
    # Initialize config
    ###################
    config = network_config_func()

    assert(isinstance(config, DefaultNetworkConfig))

    # Default settings, applied from cmd arguments.
    config.modify_classes_count(value=experiment.DataIO.LabelsScaler.classes_count())
    config.modify_learning_rate(learning_rate)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(dropout_keep_prob)
    config.modify_bag_size(BAG_SIZE)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.modify_embedding_dropout_keep_prob(1.0)
    config.modify_terms_per_context(terms_per_context)
    config.modify_use_entity_types_in_embedding(False)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())

    # Modify config parameters. This may affect
    # the settings, already applied above!
    optionally_modify_config_for_experiment(exp_type=exp_type,
                                            model_input_type=model_input_type,
                                            config=config)

    # Modify config parameters. This may affect
    # the settings, already applied above!
    modify_config_for_model(model_name=model_name,
                            model_input_type=model_input_type,
                            config=config)

    training_engine = NetworksTrainingEngine(load_model=model_load_dir is not None,
                                             experiment=experiment,
                                             create_network_func=network_func,
                                             config=config,
                                             bags_collection_type=bags_collection_type)

    training_engine.run()
