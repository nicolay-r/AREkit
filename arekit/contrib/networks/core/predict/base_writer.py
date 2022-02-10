class BasePredictWriter(object):

    def __init__(self, target):
        self._target = target

    def write(self, title, contents_it):
        raise NotImplementedError()