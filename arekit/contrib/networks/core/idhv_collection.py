import math
from collections import OrderedDict

import numpy as np


class NetworkInputDependentVariables:

    __name_separator = '>'

    def __init__(self):
        self.__text_opinion_ids = OrderedDict()
        self.__by_param_names = OrderedDict()

    # region public methods

    def add_input_dependent_values(self, names_list, tensor_values_list, text_opinion_ids,
                                   bags_per_minibatch,
                                   bag_size):
        """
        names: list
            list of string names of related 'tensor_values'
        tensor_values: list
            list of values with shape:
              * Sample based: [len(names), bags_per_minibatch, bag_size, vector_size]
              * Bags based (labels): [len(names), bags_per_minibatch, 1, 1]
        text_opinion_ids: list
            list of ids
        """
        assert(isinstance(names_list, list) and len(names_list) > 0)
        assert(isinstance(tensor_values_list, list) and len(tensor_values_list) > 0)
        assert(isinstance(text_opinion_ids, list))
        assert(isinstance(bags_per_minibatch, int))
        assert(isinstance(bag_size, int))
        assert(len(tensor_values_list) == len(names_list))

        for name_ind, name in enumerate(names_list):
            assert(isinstance(name, str))

            if name not in self.__by_param_names:
                self.__by_param_names[name] = []
                self.__text_opinion_ids[name] = []

            if bags_per_minibatch * bag_size != len(text_opinion_ids):
                raise Exception("values_list of '{}' has size {} != {} (text_opinion_inds length)".format(
                    name, bags_per_minibatch * bag_size, len(text_opinion_ids)))

            values_list = np.array(tensor_values_list[name_ind])
            values_list = values_list.flatten()
            if len(values_list) > bags_per_minibatch:
                mbatches_count = math.trunc(len(values_list) / (bags_per_minibatch * bag_size))
                values_list = values_list.reshape([bags_per_minibatch, bag_size, mbatches_count])
            else:
                # labels.
                values_list = values_list.reshape([bags_per_minibatch, 1, 1])

            # Save only first sentence ref.
            t_ind = 0
            for b_ind in range(bags_per_minibatch):

                v = values_list[b_ind]

                value = []
                names = []
                for s_index in range(len(v)):
                    value.append(v[s_index])
                    names.append(str(text_opinion_ids[t_ind]))
                    t_ind += 1

                self.__by_param_names[name].append(value)
                self.__text_opinion_ids[name].append(self.__name_separator.join(names))

    def iter_by_parameter_values(self, param_name):
        for i, tensor_value in enumerate(self.__by_param_names[param_name]):
            yield self.__text_opinion_ids[param_name][i], tensor_value

    def iter_var_names(self):
        return iter(self.__by_param_names.keys())

    # endregion