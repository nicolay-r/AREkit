class Token:
    """
    Token that stores original and resulted token values
    i.e.: term=',', token_value='<[COMMA]>'
    """
    def __init__(self, term, token_value):
        assert(isinstance(term, unicode))
        assert(isinstance(token_value, unicode))
        self.__term = term
        self.__token_value = token_value

    def get_original_value(self):
        return self.__term

    def get_token_value(self):
        return self.__token_value