import logging


logger = logging.getLogger(__name__)


class SyntaxParser:
    """
    Interface
    """

    def parse(self, text, raw_output=False, debug=False):
        """
        return: SyntaxAnalysisResult
        """
        pass


class SyntaxAnalysisResult:
    """
    Resulted tree
    """

    def __init__(self, parents, relations, terms):
        self.parents = parents
        self.relations = relations
        self.terms = terms

    def show(self):
        for i in xrange(len(self.terms)):

            s = "{} -> {} {}".format(
                '({})'.format(self.terms[i].encode('utf-8')),
                '({})'.format(self.parents[i]),
                '[{}]'.format(self.relations[i].encode('utf-8')))

            logger.info(s)
