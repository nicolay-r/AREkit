#!/usr/bin/python
import zipfile
from core.source.ruattitudes.reader import RuAttitudesFormatReader
from core.source.ruattitudes.io_utils import RuAttitudesIO

# open collection

with zipfile.ZipFile(RuAttitudesIO.get_filepath(), "r") as zip_ref:
    with zip_ref.open(RuAttitudesIO.get_collection_filepath()) as c_file:

        # create reader instance
        reader = RuAttitudesFormatReader()

        # iterating through collection
        for news in reader.iter_news(c_file):
            print "News:", news.NewsIndex
            for sentence in news.iter_sentences():
                # text
                print u" ".join(sentence.ParsedText.Terms).encode('utf-8')
                # objects
                print ",".join([object.get_value() for object in sentence.iter_objects()])
                # attitudes
                for ref_opinion in sentence.iter_ref_opinions():
                    src, target = sentence.get_objects(ref_opinion)
                    s = u"{}->{} ({})".format(src.get_value(),
                                              target.get_value(),
                                              str(ref_opinion.Sentiment.to_int())).encode('utf-8')
                    print(s)
