import logging
from os.path import join

import numpy as np

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.core.ctx_predict_log import NetworkInputDependentVariables

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


########################################
# Predefined Hidden parameters templates
########################################
__hiddenParams_template = 'hparams_{}_e{}'
__input_dependent_params_template = 'idparams_{data}_e{epoch_index}'


def save_model_hidden_values(log_dir, model, epoch_index, save_hidden_parameters):

    if not save_hidden_parameters:
        return

    names, values = model.get_hidden_parameters()

    assert (isinstance(names, list))
    assert (isinstance(values, list))
    assert (len(names) == len(values))

    for value_index, name in enumerate(names):
        variable_path = join(log_dir, __hiddenParams_template.format(name, epoch_index))
        logger.info("Save hidden values: {}".format(variable_path))
        create_dir_if_not_exists(variable_path)
        np.save(variable_path, values[value_index])


def save_minibatch_all_input_dependent_hidden_values(log_dir, data_type, epoch_index, predict_log):
    assert(isinstance(predict_log, NetworkInputDependentVariables))

    for var_name in predict_log.iter_var_names():
        __save_minibatch_variable_values(log_dir=log_dir,
                                         data_type=data_type,
                                         epoch_index=epoch_index,
                                         predict_log=predict_log,
                                         var_name=var_name)


def __save_minibatch_variable_values(log_dir, data_type, epoch_index, predict_log, var_name):
    assert(isinstance(log_dir, str))
    assert(isinstance(predict_log, NetworkInputDependentVariables))
    assert(isinstance(var_name, str))

    filename = __input_dependent_params_template.format(data='{}-{}'.format(var_name, data_type),
                                                        epoch_index=epoch_index)

    vars_path = join(log_dir, filename)
    create_dir_if_not_exists(vars_path)

    msg = "Save input dependent hidden values in a list using np.savez: {}".format(vars_path)
    logger.info(msg)

    id_and_value_pairs = list(predict_log.iter_by_parameter_values(var_name))
    id_and_value_pairs = sorted(id_and_value_pairs, key=lambda pair: pair[0])
    np.savez(vars_path, [pair[1] for pair in id_and_value_pairs])
