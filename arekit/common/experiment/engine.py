import collections


class ExperimentEngine(object):
    """ Represents a CV-Based Experiment engine
        The core method is `run`, which assumes to perform iteration per every
        iteration in and runs handler during the latter.
    """

    def __init__(self):
        self.__handlers = None

    def __call_all_handlers(self, call_func):
        assert(callable(call_func))

        if self.__handlers is None:
            return

        for handler in self.__handlers:
            call_func(handler)

    # region protected methods

    def _handle_iteration(self, iter_index):
        self.__call_all_handlers(lambda callback: callback.on_iteration(iter_index))

    def _before_running(self):
        """ Optional method that allows to implement actions before experiment iterations.
        """
        self.__call_all_handlers(lambda callback: callback.on_before_iteration())

    def _after_running(self):
        """ Optional method that allows to implement actions after experiment iterations.
        """
        self.__call_all_handlers(lambda callback: callback.on_after_iteration())

    # endregion

    def run(self, states_iter, handlers=None):
        """ Running cv_index iteration and calling handler during every iteration.
            states_iter: any iterator
        """
        assert(isinstance(states_iter, collections.Iterable))

        self.__handlers = handlers
        self._before_running()
        for iter_index, _ in enumerate(states_iter):
            self._handle_iteration(iter_index)
        self._after_running()
