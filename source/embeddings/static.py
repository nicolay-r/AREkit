import hashlib

from core.runtime.utils import get_random_vector
from core.source.embeddings.base import Embedding


class StaticEmbedding(Embedding):

    def __init__(self, vector_size):
        assert(isinstance(vector_size, int))
        super(StaticEmbedding, self).__init__()
        self.word_indices = {}
        self.vectors = []
        self.vector_size = vector_size

    def from_file(cls, filepath, binary, stemmer=None, pos_tagger=None):
        raise Exception("Unavailable for this type of embeddings")

    def create_and_add_embedding(self, word):
        assert(isinstance(word, unicode))
        assert(word not in self.word_indices)

        h = hashlib.md5(word.encode('utf-8'))
        seed = int(h.hexdigest(), 16) % 2**32
        vector = get_random_vector(self.vector_size, int(seed))
        index = len(self.vectors)

        self.word_indices[word] = index
        self.vectors.append(vector)

        return vector

    @property
    def VectorSize(self):
        return self.vector_size

    @property
    def vocab(self):
        for word, index in self.word_indices.iteritems():
            yield word, index

    def get_vector_by_index(self, index):
        assert(isinstance(index, int))
        return self.vectors[index]

    def get_word_by_index(self, index):
        assert(isinstance(index, int))
        assert(isinstance(self.word_indices, dict))
        for word, i in self.word_indices.iteritems():
            if i == index:
                return word
        raise Exception("Word has not been found")

    def find_index_by_word(self, word):
        assert(isinstance(word, unicode))
        return self.word_indices[word]

    def __contains__(self, word):
        assert(isinstance(word, unicode))
        return word in self.word_indices

    def __getitem__(self, word):
        assert(isinstance(word, unicode))
        index = self.word_indices[word]
        return self.vectors[index]

    def __len__(self):
        return len(self.vectors)


