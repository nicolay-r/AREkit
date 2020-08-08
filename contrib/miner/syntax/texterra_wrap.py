import texterra
import logging
from texterra.syntaxtree import SyntaxTree
from arekit.contrib.miner.syntax import SyntaxParser, SyntaxAnalysisResult


logger = logging.getLogger(__name__)


class TexterraSyntaxParser(SyntaxParser):

    default_host = "localhost"
    default_port = "8082"

    def __init__(self, host=None, debug=False):
        default_url = 'http://{}:{}/texterra/'.format(
            self.default_host,
            self.default_port)

        url = default_url if host is None else host

        if debug:
            logger.info("Connecting to Texterra server: {}".format(url))

        self.t = texterra.API(host=url)

    def parse(self, text, debug=False):
        """
        Parse (only for russian texts)
        text: unicode
        return: SyntaxAnalysisResult
        """
        assert(isinstance(text, unicode))
        parsed = self.t.syntax_detection(text)

        parents = []
        relations = []
        terms = []

        for tree in parsed:
            assert(isinstance(tree, SyntaxTree))

            i = 1   # Skip root token

            for label, head in zip(tree.get_labels(), tree.get_heads()):
                terms.append(tree.tokens[i])
                parents.append(head)
                relations.append(label)
                i += 1

        result = SyntaxAnalysisResult(parents, relations, terms)

        if debug:
            result.show()

        return result

