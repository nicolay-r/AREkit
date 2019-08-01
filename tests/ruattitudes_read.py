#!/usr/bin/python
from core.processing.lemmatization.mystem import MystemWrapper
from core.source.ruattitudes.reader import RuAttitudesFormatReader


stemmer = MystemWrapper()
reader = RuAttitudesFormatReader()

# iterating through collection
for news in reader.iter_news(stemmer):
    print "News:", news.NewsIndex
    for sentence in news.iter_sentences():
        # text
        print u" ".join(sentence.ParsedText.Terms).encode('utf-8')
        # objects
        print u",".join([object.get_value() for object in sentence.iter_objects()])
        # attitudes
        for ref_opinion in sentence.iter_ref_opinions():
            src, target = sentence.get_objects(ref_opinion)
            s = u"{}->{} ({})".format(src.get_value(),
                                      target.get_value(),
                                      str(ref_opinion.Sentiment.to_int())).encode('utf-8')
            print(s)
