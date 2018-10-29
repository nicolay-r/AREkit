import socket
import sys


class SyntaxNetParserWrapper:
    """
    Interface
    """
    host = "localhost"
    port = 8111

    def __init__(self):
        pass

    def parse(self, text, sentences=None, raw_output=False, debug=False):
        raw_input_s = self._prepare_raw_input_for_syntaxnet(text,
                                                            sentences)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.sendall(raw_input_s)
        raw_output_s = self._read_all_from_socket(sock)
        sock.close()

        if not raw_output_s:
            return None

        if raw_output:
            return raw_output_s

        # TODO: add output processing.
        pass

    def _prepare_raw_input_for_syntaxnet(self, text, sentences):
        """
        (C) IINemo
        Original: https://github.com/IINemo/syntaxnet_wrapper
        """
        raw_input_s = u''
        if not sentences:
            raw_input_s = text + u'\n\n'
        else:
            for sent in sentences:
                line = u' '.join((text[e.begin: e.end] for e in sent))
                raw_input_s += line
                raw_input_s += u'\n'
            raw_input_s += u'\n'

        return raw_input_s.encode('utf8')

    def _read_all_from_socket(self, sock):
        """
        (C) IINemo
        Original: https://github.com/IINemo/syntaxnet_wrapper
        """
        buf = str()

        try:
            while True:
                data = sock.recv(51200)
                if data:
                    buf += data
                else:
                    break
        except socket.error as err:
            print >> sys.stderr, 'Err: Socket error: ', err

        return buf


class PennPOSTags:
    """
    https://cs.nyu.edu/grishman/jet/guide/PennPOS.html
    """

    tags = ["CC Coordinating conjunction",
        "CD Cardinal number",
        "DT Determiner",
        "EX Existential there",
        "FW Foreign word",
        "IN Preposition or subordinating conjunction",
        "JJ Adjective",
        "JJR Adjective, comparative",
        "JJS Adjective, superlative",
        "LS List item marker",
        "MD Modal",
        "NN Noun, singular or mass",
        "NNS Noun, plural",
        "NNP Proper noun, singular",
        "NNPS Proper noun, plural",
        "PDT Predeterminer",
        "POS Possessive ending",
        "PRP Personal pronoun",
        "PRP$ Possessive pronoun",
        "RB Adverb",
        "RBR Adverb, comparative",
        "RBS Adverb, superlative",
        "RP Particle",
        "SYM Symbol",
        "TO to",
        "UH Interjection",
        "VB Verb, base form",
        "VBD Verb, past tense",
        "VBG Verb, gerund or present participle",
        "VBN Verb, past participle",
        "VBP Verb, non - 3 rd person singular present",
        "VBZ Verb, 3 rd person singular present",
        "WDT Wh - determiner",
        "WP Wh - pronoun",
        "WP$ Possessive wh - pronoun",
        "WRB Wh - adverb"]
