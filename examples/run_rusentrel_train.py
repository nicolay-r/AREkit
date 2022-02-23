import argparse

from arekit.common.experiment.api.ctx_training import ExperimentTrainingContext
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.types import FoldingType
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.callback.hidden import HiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.networks.np_utils.writer import NpzDataWriter
from arekit.contrib.networks.handlers.training import NetworksTrainingIterationHandler
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService
from examples.network.args import const

from examples.network.args.common import DistanceInTermsBetweenAttitudeEndsArg, ExperimentTypeArg, LabelsCountArg, \
    StemmerArg, TermsPerContextArg, ModelNameArg, VocabFilepathArg, ModelLoadDirArg, UseBalancingArg, \
    EmbeddingMatrixFilepathArg
from examples.network.args.const import BAG_SIZE, NEURAL_NETWORKS_TARGET_DIR
from examples.network.args.train import BagsPerMinibatchArg, DropoutKeepProbArg, EpochsCountArg, LearningRateArg, \
    ModelInputTypeArg, ModelNameTagArg
from examples.network.common import create_bags_collection_type, create_network_model_io
from examples.rusentrel.common import Common
from examples.rusentrel.config_setups import optionally_modify_config_for_experiment, modify_config_for_model
from examples.rusentrel.exp_io import CustomRuSentRelNetworkExperimentIO


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training script for obtaining Tensorflow based states, "
                                                 "based on the RuSentRel and RuAttitudes datasets (optionally)")

    # Composing cmd arguments.
    LabelsCountArg.add_argument(parser, default=3)
    ExperimentTypeArg.add_argument(parser, default="rsr")
    StemmerArg.add_argument(parser, default="mystem")
    BagsPerMinibatchArg.add_argument(parser, default=const.BAGS_PER_MINIBATCH)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser, default=None)
    ModelInputTypeArg.add_argument(parser, default=ModelInputType.SingleInstance)
    ModelNameArg.add_argument(parser, default=ModelNames.PCNN.value)
    ModelNameTagArg.add_argument(parser, default=ModelNameTagArg.NO_TAG)
    DropoutKeepProbArg.add_argument(parser, default=0.5)
    LearningRateArg.add_argument(parser, default=0.1)
    EpochsCountArg.add_argument(parser, default=150)
    VocabFilepathArg.add_argument(parser, default=None)
    EmbeddingMatrixFilepathArg.add_argument(parser, default=None)
    ModelLoadDirArg.add_argument(parser, default=None)
    UseBalancingArg.add_argument(parser, default=True)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    exp_type = ExperimentTypeArg.read_argument(args)
    labels_count = LabelsCountArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    model_input_type = ModelInputTypeArg.read_argument(args)
    model_name = ModelNameArg.read_argument(args)
    embedding_matrix_filepath = EmbeddingMatrixFilepathArg.read_argument(args)
    vocab_filepath = VocabFilepathArg.read_argument(args)
    dropout_keep_prob = DropoutKeepProbArg.read_argument(args)
    bags_per_minibatch = BagsPerMinibatchArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    learning_rate = LearningRateArg.read_argument(args)
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    model_name_tag = ModelNameTagArg.read_argument(args)
    epochs_count = EpochsCountArg.read_argument(args)
    model_load_dir = ModelLoadDirArg.read_argument(args)
    use_balancing = UseBalancingArg.read_argument(args)

    # Utilize predefined versions and folding format.
    rusentrel_version = RuSentRelVersions.V11
    ra_version = None if exp_type == ExperimentTypes.RuSentRel else RuAttitudesVersions.V20LargeNeut
    folding_type = FoldingType.Fixed
    model_target_dir = NEURAL_NETWORKS_TARGET_DIR

    # Init handler.
    bags_collection_type = create_bags_collection_type(model_input_type=model_input_type)
    network_func, network_config_func = create_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=model_input_type)

    labels_scaler = Common.create_labels_scaler(labels_count)

    exp_name = Common.create_exp_name(rusentrel_version=rusentrel_version,
                                      ra_version=ra_version,
                                      folding_type=folding_type)

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=use_balancing,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    data_folding = Common.create_folding(
        rusentrel_version=rusentrel_version,
        ruattitudes_version=ra_version,
        doc_id_func=lambda doc_id: Common.ra_doc_id_func(doc_id=doc_id))

    # Creating experiment
    exp_ctx = ExperimentTrainingContext(labels_count=labels_scaler.LabelsCount,
                                        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix),
                                        data_folding=data_folding)

    exp_io = CustomRuSentRelNetworkExperimentIO(exp_ctx)

    full_model_name = Common.create_full_model_name(model_name=model_name,
                                                    input_type=model_input_type)

    model_io = create_network_model_io(full_model_name=full_model_name,
                                       source_dir=model_load_dir,
                                       target_dir=model_target_dir,
                                       embedding_filepath=embedding_matrix_filepath,
                                       vocab_filepath=vocab_filepath,
                                       model_name_tag=model_name_tag)

    # Setup model io.
    exp_ctx.set_model_io(model_io)

    ###################
    # Initialize config
    ###################
    config = network_config_func()

    assert(isinstance(config, DefaultNetworkConfig))

    # Default settings, applied from cmd arguments.
    config.modify_classes_count(value=labels_count)
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

    data_writer = NpzDataWriter()

    nework_callbacks = [
        TrainingLimiterCallback(train_acc_limit=0.99),
        TrainingStatProviderCallback(),
        HiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer),
        InputHiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer)
    ]

    training_handler = NetworksTrainingIterationHandler(
        load_model=model_load_dir is not None,
        exp_ctx=exp_ctx,
        exp_io=exp_io,
        create_network_func=network_func,
        config=config,
        bags_collection_type=bags_collection_type,
        network_callbacks=nework_callbacks,
        training_epochs=epochs_count)

    engine = ExperimentEngine(exp_ctx.DataFolding)

    engine.run(handlers=[training_handler])
