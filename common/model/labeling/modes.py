class LabelCalculationMode:

    FIRST_APPEARED = u'take_first_appeared'
    AVERAGE = u'average'

    @staticmethod
    def supported(value):
        for s in LabelCalculationMode.iter_supported():
            if s == value:
                return True
        return False

    @staticmethod
    def iter_supported():
        for var_name in dir(LabelCalculationMode):
            yield getattr(LabelCalculationMode, var_name)