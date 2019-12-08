import logging
import socket
import sys
from arekit.processing.text.parsed import ParsedText
from syntaxnet.annot import Word
from syntaxnet.conll import ConllFormatStreamParser


logger = logging.getLogger(__name__)


class SyntaxNetParserWrapper:
    """
    (C) IINemo
    Wrapper for SyntaxNet Parser.
    """
    host = "localhost"
    port = 8111

    def __init__(self):
        pass

    def parse(self, parsed_sentence, raw_output=False, debug=False):
        assert(isinstance(parsed_sentence, ParsedText))

        parsed_sentence.unhide_token_values()
        text = u" ".join(parsed_sentence.Terms)

        if debug:
            logger.info("------------------")
            logger.info("SyntaxNetText (parsed text): {}".format(text.encode('utf-8')))
            logger.info("------------------")

        raw_input_s = self.__prepare_raw_input_for_syntaxnet(text=text,
                                                             sentences=None)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.sendall(raw_input_s)
        raw_output_s = self.__read_all_from_socket(sock)
        sock.close()

        if not raw_output_s:
            return None

        if raw_output:
            return raw_output_s

        trees = SyntaxNetParserWrapper.__parse_conll_format(text, raw_output_s)
        return trees

    # region private methods

    @staticmethod
    def __prepare_raw_input_for_syntaxnet(text, sentences):
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

    @staticmethod
    def __read_all_from_socket(sock):
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

    @staticmethod
    def __parse_conll_format(text, string):
        sent = ConllFormatStreamParser(string).next()
        words = list()
        end = 0
        for word in sent:
            word_form = word[1].decode('utf8')
            begin = text.find(word_form, end)
            end = begin + len(word_form)

            new_word = Word(begin=begin,
                            end=end,
                            word_form=word_form,
                            pos_tag=word[3].decode('utf8'),
                            morph=word[5].decode('utf8'),
                            parent=int(word[6])-1,
                            link_name=word[7].decode('utf8'))

            words.append(new_word)

        return words

    # endregion
