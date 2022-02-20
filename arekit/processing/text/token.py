class Token:
    """
    Token that stores original and resulted token values
    i.e.: term=',', token_value='<[COMMA]>'
    """
    def __init__(self, term, token_value):
        assert(isinstance(term, str))
        assert(isinstance(token_value, str))
        self.__meta_value = term
        self.__token_value = token_value

    def get_meta_value(self):
        return self.__meta_value

    def get_token_value(self):
        return self.__token_value