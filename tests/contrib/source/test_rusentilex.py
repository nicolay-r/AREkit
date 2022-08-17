import sys

from arekit.contrib.utils.lexicons.rusentilex import RuSentiLexLexicon

sys.path.append('../../../../')

lexicon = RuSentiLexLexicon.from_zip()
for term in lexicon:
    print(term)

print('порядочный' in lexicon)
