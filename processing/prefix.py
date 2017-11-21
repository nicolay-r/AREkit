# -*- coding: utf-8 -*-
import io


class SentimentPrefixProcessor:
    """ Tool that searches and repace founded constructions with sentiment
        prefixes '+' and '-'
    """

    def __init__(self, prefix_patterns):
        self.prefix_patterns = prefix_patterns

    @staticmethod
    def from_file(filepath):
        patterns = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split(',')
                pattern = args[0].strip()
                sign = args[1].strip()

                assert(pattern not in patterns)
                assert(sign == '+' or sign == '-')

                patterns[pattern] = sign

        return SentimentPrefixProcessor(patterns)

    # TODO: might be moved, so actually it's not a processor and just a
    # collection.
    def process(self, lemmas):
        """ Returns: list of lemmas
                 filtered list with replaced by '+'/'-' patterns.
        """

        assert(type(lemmas) == list)
        to_remove = []
        i = 0

        while i < len(lemmas) - 1:
            # bigram
            bigram = "%s_%s" % (lemmas[i], lemmas[i + 1])
            if bigram in self.prefix_patterns:
                lemmas.insert(i, self.prefix_patterns[bigram])
                to_remove.append(i + 1)
                to_remove.append(i + 2)
                i += 3
                continue

            # unigram
            unigram = lemmas[i]
            if unigram in self.prefix_patterns:
                lemmas.insert(i, self.prefix_patterns[unigram])
                to_remove.append(i+1)
                i += 2
                continue

            # skip
            i += 1

        # remove marked lemmas from list
        return [lemmas[j] for j in range(len(lemmas)) if j not in to_remove]
