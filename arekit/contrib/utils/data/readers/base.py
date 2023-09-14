class BaseReader(object):

    def extension(self):
        raise NotImplementedError()

    def read(self, target):
        raise NotImplementedError()
