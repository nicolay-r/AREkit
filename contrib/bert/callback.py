class Callback(object):

    def set_iter_index(self, it_index):
        raise NotImplementedError()

    def set_log_dir(self, target_dir):
        raise NotImplementedError()

    def write_results(self, result, data_type, epoch_index):
        raise NotImplementedError()

    def __enter__(self):
        """ Utilized for a single iteration.
        """
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Leaving single experiment iteration.
        """
        raise NotImplementedError()
