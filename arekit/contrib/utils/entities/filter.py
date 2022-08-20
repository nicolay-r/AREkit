class EntityFilter(object):

    def __init__(self):
        pass

    def is_ignored(self, entity, e_type):
        raise NotImplementedError()
