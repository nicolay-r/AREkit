from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.nerel import labels


class NerelAnyLabelFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {
            "OPINION_BELONGS_TO": labels.OpinionBelongsTo,
            "OPINION_RELATES_TO": labels.OpinionRelatesTo,
            "NEG_EFFECT_FROM": labels.NegEffectFrom,
            "POS_EFFECT_FROM": labels.PosEffectFrom,
            "NEG_STATE_FROM": labels.NegStateFrom,
            "POS_STATE_FROM": labels.PosStateFrom,
            "NEGATIVE_TO": labels.NegativeTo,
            "POSITIVE_TO": labels.PositiveTo,
            "STATE_BELONGS_TO": labels.STATE_BELONGS_TO,
            "POS_AUTHOR_FROM": labels.PosAuthorFrom,
            "NEG_AUTHOR_FROM": labels.NegAuthorFrom,
            "ALTERNATIVE_NAME": labels.ALTERNATIVE_NAME,
            "ORIGINS_FROM": labels.ORIGINS_FROM,
            "START_TIME": labels.START_TIME,
            "OWNER_OF": labels.OWNER_OF,
            "SUBEVENT_OF": labels.SUBEVENT_OF,
            "PARENT_OF": labels.PARENT_OF,
            "SUBORDINATE_OF": labels.SUBORDINATE_OF,
            "PART_OF": labels.PART_OF,
            "TAKES_PLACE_IN": labels.TAKES_PLACE_IN,
            "PARTICIPANT_IN": labels.PARTICIPANT_IN,
            "WORKPLACE": labels.WORKPLACE,
            "PENALIZED_AS": labels.PENALIZED_AS,
            "WORKS_AS": labels.WORKS_AS,
            "PLACE_OF_DEATH": labels.PLACE_OF_DEATH,
            "PLACE_OF_BIRTH": labels.PLACE_OF_BIRTH,
            "HAS_CAUSE": labels.HAS_CAUSE,
            "AWARDED_WITH": labels.AWARDED_WITH,
            "CAUSE_OF_DEATH": labels.CAUSE_OF_DEATH,
            "CONVICTED_OF": labels.CONVICTED_OF,
            "DATE_DEFUNCT_IN": labels.DATE_DEFUNCT_IN,
            "DATE_FOUNDED_IN": labels.DATE_FOUNDED_IN,
            "DATE_OF_BIRTH": labels.DATE_OF_BIRTH,
            "DATE_OF_CREATION": labels.DATE_OF_CREATION,
            "DATE_OF_DEATH": labels.DATE_OF_DEATH,
            "END_TIME": labels.END_TIME,
            "EXPENDITURE": labels.EXPENDITURE,
            "FOUNDED_BY": labels.FOUNDED_BY,
            "KNOWS": labels.KNOWS,
            "RELATIVE": labels.RELATIVE,
            "LOCATED_IN": labels.LOCATED_IN,
            "RELIGION_OF": labels.RELIGION_OF,
            "MEDICAL_CONDITION": labels.MEDICAL_CONDITION,
            "SCHOOLS_ATTENDED": labels.SCHOOLS_ATTENDED,
            "MEMBER_OF": labels.MEMBER_OF,
            "SIBLING": labels.SIBLING,
            "ORGANIZES": labels.ORGANIZES,
            "SPOUSE": labels.SPOUSE
        }

        super(NerelAnyLabelFormatter, self).__init__(stol=stol)
