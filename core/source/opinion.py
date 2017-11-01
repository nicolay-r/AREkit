# -*- coding: utf-8 -*-
import io
import core.environment as env


class OpinionCollection:
    """ Collection of sentiment opinions between entities
    """

    def __init__(self, opinions=None):
        self.opinions = [] if opinions is None else opinions
        self.unique = set()

        for o in self.opinions:
            self.unique.add(self._get_opinion_key(o.entity_left, o.entity_right))

    @staticmethod
    def from_file(filepath):
        """ Read opinion collection from file
        """
        opinions = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                args = line.strip().split(',')

                if len(args) < 4:
                    print "not enough arguments at line: {}, '{}'".format(i, line.encode('utf-8'))
                    continue

                entity_left = args[0].strip().lower()
                entity_right = args[1].strip().lower()

                o = Opinion(entity_left,
                            entity_right,
                            args[2].strip(),
                            args[3].strip())
                opinions.append(o)

        return OpinionCollection(opinions)

    def has_opinion(self, entity_left, entity_right, lemmatize=False):
        return self._get_opinion_key(entity_left, entity_right) in self.unique

    def add_opinion(self, opinion):
        self.opinions.append(opinion)
        self.unique.add(self._get_opinion_key(
            opinion.entity_left, opinion.entity_right))

    @staticmethod
    def _get_opinion_key(l_value, r_value):
        return "{}_{}".format(
            env.stemmer.lemmatize_to_str(l_value).encode('utf-8'),
            env.stemmer.lemmatize_to_str(r_value).encode('utf-8'))

    def limit(self, count):
        # this is incorrect. temprorary
        self.opinions = self.opinions[:count]

    def save(self, filepath):
        with io.open(filepath, 'w') as f:
            for o in self.opinions:
                if o.sentiment != unicode('neu', encoding='utf-8'):
                    f.write(o.to_unicode())
                    f.write(unicode("\n"))

    def __iter__(self):
        for a in self.opinions:
            yield a


class Opinion:
    """ Source opinion description
    """

    def __init__(self, entity_left, entity_right, sentiment, time):
        assert(type(entity_left) == unicode)
        assert(type(entity_right) == unicode)
        assert(type(sentiment) == unicode)
        assert(type(time) == unicode)
        self.entity_left = entity_left
        self.entity_right = entity_right
        self.sentiment = sentiment
        self.time = time

    def to_unicode(self):
        return "{}, {}, {}, {}".format(
                self.entity_left.encode('utf-8'),
                self.entity_right.encode('utf-8'),
                self.sentiment.encode('utf-8'),
                self.time.encode('utf-8')).decode('utf-8')
