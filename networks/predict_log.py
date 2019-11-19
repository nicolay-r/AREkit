from collections import OrderedDict


class NetworkInputDependentVariables:

    def __init__(self):
        self.__text_opinion_ids = OrderedDict()
        self.__by_param_names = OrderedDict()

    # region public methods

    def add_input_dependent_values(self, names_list, tensor_values_list, text_opinion_ids):
        """
        names: list
            list of string names of related 'tensor_values'
        tensor_values: list
            list of values with shape [len(text_opinion_ids)]
        text_opinion_ids: list
            list of ids
        """
        assert(isinstance(names_list, list) and len(names_list) > 0)
        assert(isinstance(tensor_values_list, list) and len(tensor_values_list) > 0)
        assert(isinstance(text_opinion_ids, list))
        assert(len(tensor_values_list) == len(names_list))

        for name_ind, name in enumerate(names_list):
            assert(isinstance(name, unicode))

            if name not in self.__by_param_names:
                self.__by_param_names[name] = []
                self.__text_opinion_ids[name] = []

            values_list = tensor_values_list[name_ind]

            if len(values_list) != len(text_opinion_ids):
                raise Exception("values_list of '{}' has size {} != {} (text_opinion_inds length)".format(
                    name, len(values_list), len(text_opinion_ids)))

            for text_opinion_ind, id in enumerate(text_opinion_ids):
                self.__by_param_names[name].append([text_opinion_ind])
                self.__text_opinion_ids[name].append(id)

    def iter_by_parameter_values(self, param_name):
        for i, tensor_value in enumerate(self.__by_param_names[param_name]):
            yield self.__text_opinion_ids[param_name][i], tensor_value

    def iter_var_names(self):
        return self.__by_param_names.iterkeys()

    # endregion
