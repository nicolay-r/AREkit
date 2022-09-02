class BaseSamplesIO(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    @property
    def Reader(self):
        raise NotImplementedError()

    @property
    def Writer(self):
        """ For serialization
        """
        raise NotImplementedError()

    def create_target(self, data_type, data_folding):
        """ Path for reaiding/viewing
        """
        raise NotImplementedError()
