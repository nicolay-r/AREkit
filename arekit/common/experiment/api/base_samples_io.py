class BaseSamplesIO(object):
    """ Represents base experiment utils for input/output for:
        samples -- data that utilized for experiments;
        results -- evaluation of experiments.
    """

    def create_view(self, data_type, data_folding):
        """ For viewing/reading
        """
        raise NotImplementedError()

    def create_writer(self):
        """ For serialization
        """
        raise NotImplementedError()

    def create_target(self, data_type, data_folding):
        """ Path for reaiding/viewing
        """
        raise NotImplementedError()
