# Source

## Embeddings

Represents a wrapper over [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) model api of [gensim](https://radimrehurek.com/gensim/) library.
This core provides an additional wrappers for:
* News collection from [rusvectores](http://rusvectores.org/ru/models/), which has specific pos prefixes for words of vocabulary;
* Wrapper for additional punctuation signs (tokens) in text, i.e. `":", ";", ".", "!"` etc.
