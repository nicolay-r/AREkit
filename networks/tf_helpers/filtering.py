import tensorflow as tf


def filter_batch_elements(elements, inds, handler, elements_type=tf.int32):
    """
    Process each element in batch (filter) by handler function
    elements:  [batch_size, terms_per_context]
    """
    batch_size = elements.shape[0]

    filtered = tf.TensorArray(
        dtype=elements_type,
        name="context_iter",
        size=batch_size,
        infer_shape=False,
        dynamic_size=True)

    _, _, _, filtered = tf.while_loop(
        lambda i, *_: tf.less(i, batch_size),
        handler,
        (0, elements, inds, filtered))

    return filtered.stack()


def select_entity_related_elements(i, elements, inds, filtered):
    """
    elements: [batch, terms_per_context]
    inds: [batch, terms_per_context]
    """

    row_elements = tf.squeeze(tf.gather(elements, [i], axis=0))
    row_inds = tf.squeeze(tf.gather(inds, [i], axis=0))

    result = tf.gather(row_elements, row_inds)   # row: [entities_per_context]

    return (i + 1,
            elements,
            inds,
            filtered.write(i, tf.squeeze(result)))


