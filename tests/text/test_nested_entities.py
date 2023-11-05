import unittest

from arekit.common.text.parsed import BaseParsedText
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.entities_default import TextEntitiesParser


class TestNestedEntities(unittest.TestCase):

    def test(self):
        s = """24 марта президент [США] [Джо-Байден] провел переговоры с
               лидерами стран [Евросоюза] в [Брюсселе] , вызвав внимание рынка и предположения о
               том, что [Америке] удалось уговорить [ЕС] совместно бойкотировать российские нефть
               и газ.  [[Европейский]-[Союз]] крайне зависим от [России] в плане поставок нефти и
               газа."""

        tep = TextEntitiesParser()

        text_parser = BaseTextParser(pipeline=[
            TextEntitiesParser(),
        ])

        parsed_text = text_parser.run(s.split())
        assert(isinstance(parsed_text, BaseParsedText))
        print(parsed_text._terms)


if __name__ == '__main__':
    unittest.main()
