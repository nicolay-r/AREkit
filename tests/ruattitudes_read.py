#!/usr/bin/python
import zipfile
from reader.ruattitudes import RuAttitudesFormatReader


# open collection
with zipfile.ZipFile("collection.zip", "r") as zip_ref:
    with zip_ref.open("collection.txt") as c_file:

        # create reader instance
        reader = RuAttitudesFormatReader()

        # iterating through collection
        for news in reader.iter_news(c_file):
            print("News:", news.NewsIndex)
            for sentence in news.iter_sentences():
                # text
                print(" ".join(sentence.ParsedText))
                # objects
                print(",".join([object.get_value() for object in sentence.iter_objects()]))
                # attitudes
                for ref_opinion in sentence.iter_ref_opinions():
                    src, target = sentence.get_objects(ref_opinion)
                    s = "{}->{} ({})".format(src.get_value(),
                                             target.get_value(),
                                             str(ref_opinion.Sentiment.to_int()))
                    print(s)
