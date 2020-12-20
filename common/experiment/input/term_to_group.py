from arekit.common.synonyms import SynonymsCollection


def term_to_group_func(value, synonyms):
    assert(isinstance(value, unicode))
    assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)

    if synonyms is None:
        return None

    if not synonyms.contains_synonym_value(value):
        return None

    return synonyms.get_synonym_group_index(value)
