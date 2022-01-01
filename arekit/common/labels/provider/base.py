class BasePairLabelProvider(object):

    def __init__(self):
        pass

    def provide(self, source, target):
        raise NotImplementedError()
