from arekit.common import utils
from arekit.common.embeddings.tokens import TokenEmbedding


def create_tokens_embedding(vector_size):
    assert(isinstance(vector_size, int))

    seed_token_offset = 12345
    return TokenEmbedding.from_supported_tokens(
        vector_size=vector_size,
        random_vector_func=lambda size, t_ind: utils.get_random_normal_distribution(
            vector_size=size,
            seed=t_ind + seed_token_offset,
            loc=0.05,
            scale=0.025))
