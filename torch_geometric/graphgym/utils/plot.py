import os.path as osp


def view_emb(emb, dir):
    """Visualize a embedding matrix.

    Args:
        emb (torch.tensor): Embedding matrix with shape (N, D). D is the
        feature dimension.
        dir (str): Output directory for the embedding figure.
    """
    # Import plotting libraries lazily and guard against import-time warnings
    # or missing optional dependencies so importing this module during test
    # collection does not trigger side-effects.
    import importlib
    import warnings

    try:
        with warnings.catch_warnings():
            # Suppress deprecation/warning messages that can be emitted by
            # underlying libraries (e.g., pyparsing) during import.
            warnings.simplefilter('ignore')
            plt = importlib.import_module('matplotlib.pyplot')
            sns = importlib.import_module('seaborn')
            PCA = importlib.import_module('sklearn.decomposition').PCA
    except Exception:
        # Plotting libraries are not available or failed to import; skip plotting.
        return

    try:
        sns.set_context('poster')
    except Exception:
        # If seaborn fails to set context for any reason, continue without it.
        pass

    if emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb = pca.fit_transform(emb)
    plt.figure(figsize=(10, 10))
    plt.scatter(emb[:, 0], emb[:, 1])
    plt.savefig(osp.join(dir, 'emb_pca.png'), dpi=100)
