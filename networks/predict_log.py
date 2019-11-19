import numpy as np
from collections import OrderedDict


class NetworkInputDependentVariables:

    __name_separator = u'>'

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
            list of values with shape [len(names), bags_per_minibatch, bag_size]
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
            assert(isinstance(name, unicode))

            if name not in self.__by_param_names:
                self.__by_param_names[name] = []
                self.__text_opinion_ids[name] = []

            if bags_per_minibatch * bag_size != len(text_opinion_ids):
                print bags_per_minibatch
                print bag_size
                raise Exception("values_list of '{}' has size {} != {} (text_opinion_inds length)".format(
                    name, bags_per_minibatch * bag_size, len(text_opinion_ids)))

            values_list = np.array(tensor_values_list[name_ind])
            values_list = values_list.reshape([bags_per_minibatch, bag_size])

            # Save only first sentence ref.
            t_ind = 0
            for b_ind in xrange(bags_per_minibatch):

                value = []
                names = []
                for s_index in xrange(bag_size):
                    value.append(values_list[b_ind][s_index])
                    names.append(str(text_opinion_ids[t_ind]))
                    t_ind += 1

                self.__by_param_names[name].append(value)
                self.__text_opinion_ids[name].append(self.__name_separator.join(names))

    def iter_by_parameter_values(self, param_name):
        for i, tensor_value in enumerate(self.__by_param_names[param_name]):
            yield self.__text_opinion_ids[param_name][i], tensor_value

    def iter_var_names(self):
        return self.__by_param_names.iterkeys()

    # endregion
