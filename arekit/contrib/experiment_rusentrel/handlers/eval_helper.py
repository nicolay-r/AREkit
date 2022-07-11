class EvalHelper(object):
    """ Specific provide for results evaluation.
    """

    def get_results_dir(self, target_dir):
        raise NotImplementedError()

    def get_results_target(self, iter_index, epoch_index):
        raise NotImplementedError()
