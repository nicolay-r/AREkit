import os
import gc
import logging

from arekit.contrib.experiments.io_utils_base import BaseExperimentsIOUtils
from arekit.contrib.experiments.nn_io.base import BaseExperimentNeuralNetworkIO
from arekit.contrib.experiments.nn_io.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.networks.callback import Callback
from arekit.networks.data_type import DataType


def run_testing(full_model_name,
                create_config,
                create_network,
                create_model,
                create_callback,
                create_nn_io,
                evaluator_class,
                experiments_io,
                cv_count=1,
                common_callback_modification_func=None,
                custom_config_modification_func=None,
                common_config_modification_func=None,
                cancel_training_by_cost=True):
    """
    :param experiments_io:
    :param full_model_name: unicode
        model name
    :param create_config: func
    :param create_network:
    :param create_model:
    :param create_callback:
    :param create_nn_io:
    :param evaluator_class:
    :param cv_count: int, cv_count > 0
        1 -- considered a fixed train/test separation.
    :param common_callback_modification_func:
    :param common_config_modification_func:
        for all models
    :param custom_config_modification_func:
        for model
    :param cancel_training_by_cost:
    """
    assert(isinstance(full_model_name, unicode))
    assert(callable(create_config))
    assert(callable(create_network))
    assert(callable(create_model))
    assert(callable(create_callback))
    assert(callable(common_callback_modification_func) or common_callback_modification_func is None)
    assert(callable(common_config_modification_func) or common_config_modification_func is None)
    assert(callable(custom_config_modification_func) or custom_config_modification_func is None)
    assert(callable(evaluator_class))
    assert(isinstance(experiments_io, BaseExperimentsIOUtils))
    assert(isinstance(cv_count, int) and cv_count > 0)
    assert(isinstance(cancel_training_by_cost, bool))

    # Disable tensorflow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Setup logging format
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)8s %(name)s | %(message)s')
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Log
    logger.info("Full-Model-Name: {}".format(full_model_name))
    logger.info("Run: Saving neutral annotations task.")
    logger.info("Initialization: Building parsed_news collection")

    for data_type in DataType.iter_supported():
        experiments_io.NeutralAnnontator.create(data_type=data_type)

    nn_io, callback = __create_nn_io_and_callback(
        cv_count=cv_count,
        experiments_io=experiments_io,
        create_nn_io_func=create_nn_io,
        create_callback_func=create_callback,
        model_name=full_model_name,
        cancel_training_by_cost=cancel_training_by_cost,
        clear_model_contents=True)

    assert(isinstance(callback, Callback))
    assert(isinstance(nn_io, RuSentRelBasedNeuralNetworkIO))

    for cv_index in range(nn_io.CVCount):

        # Initialize config
        config = create_config()
        assert(isinstance(config, DefaultNetworkConfig))
        nn_io.init_synonyms_collection(config.Stemmer)

        # Initialize network
        network = create_network()

        # Setup config
        if common_config_modification_func is not None:
            common_config_modification_func(config=config)
        if custom_config_modification_func is not None:
            custom_config_modification_func(config)

        # Setup callback
        if common_callback_modification_func is not None:
            common_callback_modification_func(callback)
        callback.set_test_on_epochs(config.TestOnEpochs)
        callback.reset_experiment_dependent_parameters()

        # Initialize model
        model = create_model(nn_io=nn_io,
                             network=network,
                             config=config,
                             evaluator_class=evaluator_class,
                             callback=callback)

        ###########
        # Run model
        ###########
        print u"Running model '{}' at cv_index {}".format(full_model_name, nn_io.CVCurrentIndex)
        model.run(load_model=False)

        del config
        del network
        del model

        nn_io.inc_cv_index()
        gc.collect()

# region private functions


def __create_nn_io_and_callback(
        cv_count,
        experiments_io,
        create_nn_io_func,
        create_callback_func,
        model_name,
        cancel_training_by_cost,
        clear_model_contents):
    assert(isinstance(cv_count, int))
    assert(isinstance(experiments_io, BaseExperimentsIOUtils))
    assert(callable(create_nn_io_func))
    assert(callable(create_callback_func))
    assert(isinstance(model_name, unicode))
    assert(isinstance(cancel_training_by_cost, bool))
    assert(isinstance(clear_model_contents, bool))

    nn_io = create_nn_io_func(model_name=model_name,
                              experiments_io=experiments_io,
                              cv_count=cv_count)

    assert(isinstance(nn_io, BaseExperimentNeuralNetworkIO))

    callback = create_callback_func(log_dir=nn_io.get_logfile_dir())

    callback.PredictVerbosePerFileStatistic = False

    return nn_io, callback

# endregion


