from collections import OrderedDict


class NetworkInputDependentVariables:

    def __init__(self):
        self.__text_opinion_ids = OrderedDict()
        self.__by_param_names = OrderedDict()

    # region public methods

    def has_variable(self, var_name):
        return var_name in self.__by_param_names

    def add(self, names, tensor_values, text_opinion_ids):
        assert(isinstance(names, list) and len(names) > 0)
        assert(isinstance(tensor_values, list) and len(tensor_values) > 0)
        assert(isinstance(text_opinion_ids, list))
        assert(len(tensor_values) == len(names))

        for i, name in enumerate(names):
            assert(isinstance(name, unicode))

            if name not in self.__by_param_names:
                self.__by_param_names[name] = []
                self.__text_opinion_ids[name] = []

            for j, id in enumerate(text_opinion_ids):
                self.__by_param_names[name].append(tensor_values[i][j])
                self.__text_opinion_ids[name].append(id)

    def iter_by_parameter_values(self, param_name):
        for i, tensor_value in enumerate(self.__by_param_names[param_name]):
            yield self.__text_opinion_ids[param_name][i], tensor_value

    def iter_var_names(self):
        return self.__by_param_names.iterkeys()

    # endregion
