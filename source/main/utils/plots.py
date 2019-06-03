import logging


def plot_attn_terminal(src, pred, weights):
    """

    :param src: list
    :param pred: list
    :param weights: numpy array dim==2. It should be a square matrix, axis 0  corresponding to pred
    :return:
    """
    no_steps = 10
    no_samples = 3
    for i in range(min(no_steps, len(src))):
        indices = weights[i].argsort()[-no_samples:]
        logging.info('Top 3 words influencing for predicted word %s: %s', pred[i],
                     [src[idx] if idx < len(src) else 'PADDING' for idx in indices])
