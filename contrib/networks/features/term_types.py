from arekit.contrib.networks.features.utils import create_zeros


def calculate_term_types(terms, entity_inds_set):
    assert(isinstance(terms, list))
    assert(isinstance(entity_inds_set, set))

    vector = create_zeros(size=len(terms))

    for t_index, term in enumerate(terms):
        if t_index in entity_inds_set:
            vector[t_index] = 1

    return vector

