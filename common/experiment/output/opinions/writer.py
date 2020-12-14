from arekit.common.utils import create_dir_if_not_exists


def save_opinion_collections(opinion_collection_iter, create_file_func, save_to_file_func, keep_doc_id_func):
    assert(callable(keep_doc_id_func))

    for doc_id, collection in opinion_collection_iter:

        filepath = create_file_func(doc_id)
        create_dir_if_not_exists(filepath)

        if not keep_doc_id_func(doc_id):
            continue

        save_to_file_func(filepath=filepath,
                          collection=collection)
