class BaseArg:

    @staticmethod
    def read_argument(args):
        raise NotImplementedError()

    @staticmethod
    def add_argument(parser):
        raise NotImplementedError()
