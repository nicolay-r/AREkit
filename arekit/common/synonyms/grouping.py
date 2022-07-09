from arekit.common.synonyms.base import SynonymsCollection


class SynonymsCollectionValuesGroupingProviders:
    """ Providers for the grouping.
    """

    @staticmethod
    def provide_existed_or_register_missed_value(synonyms, value):
        """ grouping with a potential expansion.
        """
        assert(isinstance(synonyms, SynonymsCollection))
        if not synonyms.contains_synonym_value(value):
            synonyms.add_synonym_value(value)
        return synonyms.get_synonym_group_index(value)

    @staticmethod
    def provide_existed_value(synonyms, value):
        """ grouping by using only existed value.
        """
        return synonyms.get_synonym_group_index(value)
