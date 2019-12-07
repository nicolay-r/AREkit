from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.ruattitudes.reader import RuAttitudesFormatReader

stemmer = MystemWrapper()
reader = RuAttitudesFormatReader()

ids = set()
for news in reader.iter_news(stemmer):
    assert(isinstance(news, RuAttitudesNews))
    if news.NewsIndex in ids:
        print "index already exist: {}".format(news.NewsIndex)
    ids.add(news.NewsIndex)
print "OK"
