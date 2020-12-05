from arekit.common.languages.pos import PartOfSpeechType
from arekit.contrib.networks.features.utils import create_filled_array
from arekit.processing.pos.base import POSTagger


def calculate_term_pos(terms, entity_inds_set, pos_tagger):
    assert(isinstance(terms, list))
    assert(isinstance(entity_inds_set, set))
    assert(isinstance(pos_tagger, POSTagger))

    vector = create_filled_array(size=len(terms),
                                 value=PartOfSpeechType.Unknown)

    for t_index, term in enumerate(terms):

        if t_index in entity_inds_set:
            pos = PartOfSpeechType.Unknown
        else:
            pos = pos_tagger.get_term_pos(term)

        vector[t_index] = pos

    return vector