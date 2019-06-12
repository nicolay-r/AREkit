import socket
import sys
from core.processing.text.parsed import ParsedText
from syntaxnet.annot import Word
from syntaxnet.conll import ConllFormatStreamParser


class SyntaxNetParserWrapper:
    """
    Interface
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
            print "------------------"
            print "SyntaxNetText (parsed text): {}".format(text.encode('utf-8'))
            print "------------------"

        raw_input_s = self._prepare_raw_input_for_syntaxnet(text, sentences=None)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.sendall(raw_input_s)
        raw_output_s = self._read_all_from_socket(sock)
        sock.close()

        if not raw_output_s:
            return None

        if raw_output:
            return raw_output_s

        trees = self._parse_conll_format(text, raw_output_s)
        return trees

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

    def _parse_conll_format(self, text, string):
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



