# -*- coding: cp1251 -*-
import io
from pymystem3 import Mystem

# TODO: remove this later
MYSTEM = Mystem()


class OpinionCollection:
    """ Collection of sentiment opinions between entities
    """

    def __init__(self, opinions):
        self.opinions = opinions
        self.mystem = Mystem()

    @staticmethod
    def from_file(filepath):
        """ Read opinion collection from file
        """
        opinions = []
        with io.open(filepath, "r", encoding='utf-8') as f:
            for line in f.readlines():
                args = line.split(',')
                o = Opinion(args[0].strip(), args[1].strip(),
                            args[2].strip(), args[3].strip())
                opinions.append(o)

        return OpinionCollection(opinions)

    def has_opinion(self, entity_left, entity_right, lemmatize=False):
        for o in self.opinions:
            if o.is_equal(entity_left, entity_right, lemmatize):
                return True

        return False

    def __iter__(self):
        for a in self.opinions:
            yield a


class Opinion:
    """ Opinion description
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

    # TODO: add a relation class
    def is_equal(self, entity_left, entity_right, lemmatize=False):

        i_el = MYSTEM.lemmatize(entity_left) if lemmatize else entity_left
        i_er = MYSTEM.lemmatize(entity_right) if lemmatize else entity_right
        o_el = MYSTEM.lemmatize(self.entity_left) if lemmatize else self.entity_left
        o_er = MYSTEM.lemmatize(self.entity_right) if lemmatize else self.entity_right

        return i_el == o_el and i_er == o_er

    def show(self):
        print "{}, {}, {}, {}".format(
                self.entity_left.encode('utf-8'),
                self.entity_right.encode('utf-8'),
                self.sentiment.encode('utf-8'),
                self.time.encode('utf-8'))
