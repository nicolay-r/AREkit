class ExperimentNameProvider(object):

    def __init__(self, name, suffix):
        assert(isinstance(name, str))
        assert(isinstance(suffix, str))
        self.__name = name
        self.__suffix = suffix

    def provide(self):
        # Suffix allows to provide additional experiment setups
        # In a form of a string, which might be behind the original
        # experiment implementation, such as:
        # input samples balancing usage, ditances for samples filtration, etc..
        return "{name}-{suffix}".format(name=self.__name, suffix=self.__suffix)
