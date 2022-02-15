# %%
from itertools import product

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from embeddings import SNE, TSNE

def plot_result(x_transformed, y, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
    for c, label in zip(colors, digits.target_names):
        ax.scatter(x_transformed[y == int(label), 0], x_transformed[y == int(label), 1], color=c, label=label)
        ax.legend()
        ax.set_title(title, fontsize=16)
    plt.show()


if __name__ == "__main__":
    digits = load_digits()
    digit_qty = 200
    # only use top x samples for faster computation
    X, y = digits.data[:digit_qty, :], digits.target[:digit_qty]

    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    plot_result(X_pca, y, "PCA")

    sne = SNE(n_components=2, perplexity=50, n_epochs=200, lr=0.1)
    X_sne = sne.fit_transform(X_sc)
    plot_result(X_sne, y, "SNE")

    tsne = TSNE(n_components=2, perplexity=50, n_epochs=500, lr=0.1)
    X_tsne = tsne.fit_transform(X_sc)
    plot_result(X_tsne, y, "t-SNE")