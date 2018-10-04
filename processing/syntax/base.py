class SyntaxParser:
    """
    Interface
    """

    def parse(self):
        """
        return: SyntaxAnalysisResult
        """
        pass


class SyntaxAnalysisResult:
    """
    Resulted tree
    """
    def __init__(self, heads, relations, terms):
        self.heads = heads
        self.relations = relations
        self.terms = terms

