import os
import gc
import logging

from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.contrib.experiments.rusentrel import RuSentRelBasedNeuralNetworkIO
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.networks.callback import Callback


def run_testing(full_model_name,
                create_config,
                create_network,
                create_model,
                create_experiment,
                data_io,
                cv_count=1,
                common_callback_modification_func=None,
                custom_config_modification_func=None,
                common_config_modification_func=None):
    """
    :param data_io:
    :param full_model_name: unicode
        model name
    :param create_config: func
    :param create_network:
    :param create_model:
    :param create_experiment:
    :param cv_count: int, cv_count > 0
        1 -- considered a fixed train/test separation.
    :param common_callback_modification_func:
    :param common_config_modification_func:
        for all models
    :param custom_config_modification_func:
        for model
    """
    assert(isinstance(full_model_name, unicode))
    assert(callable(create_config))
    assert(callable(create_network))
    assert(callable(create_model))
    assert(callable(common_callback_modification_func) or common_callback_modification_func is None)
    assert(callable(common_config_modification_func) or common_config_modification_func is None)
    assert(callable(custom_config_modification_func) or custom_config_modification_func is None)
    assert(isinstance(data_io, DataIO))
    assert(isinstance(cv_count, int) and cv_count > 0)

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

    # TODO. Refactor
    data_io.set_model_name(full_model_name)
    data_io.ModelIO.set_model_name(value=full_model_name)

    # TODO. Refactor.
    experiment = __create_experiment(
        data_io=data_io,
        create_experiment_func=create_experiment,
        clear_model_contents=True)

    # TODO. This should be intialized automatically somewhere else.
    data_io.CVFoldingAlgorithm.set_cv_count(cv_count)

    # Initialize data_io
    for data_type in DataType.iter_supported():
        data_io.NeutralAnnotator.create_collection(data_type=data_type)

    callback = data_io.Callback
    callback.PredictVerbosePerFileStatistic = False

    assert(isinstance(callback, Callback))
    assert(isinstance(experiment, RuSentRelBasedNeuralNetworkIO))

    for cv_index in range(data_io.CVFoldingAlgorithm.CVCount):

        data_io.CVFoldingAlgorithm.set_iteration_index(cv_index)

        # Initialize config
        config = create_config()
        assert(isinstance(config, DefaultNetworkConfig))

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

        callback.reset_experiment_dependent_parameters()

        # Initialize model
        model = create_model(experiment=experiment,
                             network=network,
                             config=config,
                             callback=callback)

        ###########
        # Run model
        ###########
        print u"Running model '{}' at cv_index {}".format(full_model_name, data_io.CVFoldingAlgorithm.IterationIndex)
        model.run_training(load_model=False,
                           epochs_count=callback.Epochs)

        del config
        del network
        del model

        gc.collect()

# region private functions


# TODO. remove __create_experiment func
def __create_experiment(
        data_io,
        create_experiment_func,
        clear_model_contents):
    assert(isinstance(data_io, DataIO))
    assert(callable(create_experiment_func))
    assert(isinstance(clear_model_contents, bool))

    return create_experiment_func(data_io=data_io,
                                  prepare_model_root=clear_model_contents)

# endregion


