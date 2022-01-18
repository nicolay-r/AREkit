import logging
import numpy as np

from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.networks.core.ctx_predict_log import NetworkInputDependentVariables

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TODO. #259 related -- move into separated InputDependentHiddenWriterCallback.
def save_minibatch_all_input_dependent_hidden_values(path_by_var_name_func, predict_log):
    assert(callable(path_by_var_name_func))
    assert(isinstance(predict_log, NetworkInputDependentVariables))

    for var_name in predict_log.iter_var_names():
        __save_minibatch_variable_values(target=path_by_var_name_func(var_name),
                                         predict_log=predict_log,
                                         var_name=var_name)


def __save_minibatch_variable_values(target, predict_log, var_name):
    assert(isinstance(predict_log, NetworkInputDependentVariables))
    create_dir_if_not_exists(target)
    id_and_value_pairs = list(predict_log.iter_by_parameter_values(var_name))
    id_and_value_pairs = sorted(id_and_value_pairs, key=lambda pair: pair[0])
    np.savez(target, [pair[1] for pair in id_and_value_pairs])
