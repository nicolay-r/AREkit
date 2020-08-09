# -*- coding: utf-8 -*-
from arekit.contrib.source.lexicons.rusentilex import RuSentiLexLexicon

lexicon = RuSentiLexLexicon.from_zip()
for term in lexicon:
    print term

print u'порядочный' in lexicon
