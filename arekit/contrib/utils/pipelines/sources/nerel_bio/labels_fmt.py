from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.nerelbio import labels


class NerelBioAnyLabelFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {
            "ACTIVITY": labels.ACTIVITY,
            "MEDPROC": labels.MEDPROC,
            "MONEY": labels.MONEY,
            "ADMINISTRATION_ROUTE": labels.ADMINISTRATION_ROUTE,
            "MENTALPROC": labels.MENTALPROC,
            "NATIONALITY": labels.NATIONALITY,
            "ANATOMY": labels.ANATOMY,
            "PHYS": labels.PHYS,
            "NUMBER": labels.NUMBER,
            "CHEM": labels.CHEM,
            "SCIPROC": labels.SCIPROC,
            "ORDINAL": labels.ORDINAL,
            "DEVICE": labels.DEVICE,
            "AGE": labels.AGE,
            "ORGANIZATION": labels.ORGANIZATION,
            "DISO": labels.DISO,
            "CITY": labels.CITY,
            "PERCENT": labels.PERCENT,
            "FINDING": labels.FINDING,
            "COUNTRY": labels.COUNTRY,
            "PERSON": labels.PERSON,
            "FOOD": labels.FOOD,
            "DATE": labels.DATE,
            "PRODUCT": labels.PRODUCT,
            "GENE": labels.GENE,
            "DISTRICT": labels.DISTRICT,
            "PROFESSION": labels.PROFESSION,
            "INJURY_POISONING": labels.INJURY_POISONING,
            "EVENT": labels.EVENT,
            "STATE_OR_PROVINCE": labels.STATE_OR_PROVINCE,
            "HEALTH_CARE_ACTIVITY": labels.HEALTH_CARE_ACTIVITY,
            "FAMILY": labels.FAMILY,
            "TIME": labels.TIME,
            "LABPROC": labels.LABPROC,
            "FACILITY": labels.FACILITY,
            "LIVB": labels.LIVB,
            "LOCATION": labels.LOCATION
        }

        super(NerelBioAnyLabelFormatter, self).__init__(stol=stol)
