from arekit.common.synonyms.base import SynonymsCollection


class SimpleSynonymCollection(SynonymsCollection):

    def __init__(self, iter_group_values_lists, is_read_only, debug):
        super(SimpleSynonymCollection, self).__init__(iter_group_values_lists=iter_group_values_lists,
                                                      is_read_only=is_read_only,
                                                      debug=debug)

    def _create_external_sid(self, value):
        return value

    def _create_internal_sid(self, value):
        return value
