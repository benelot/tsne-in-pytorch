
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class SNE:
    """
    Stochastic Neighbor Embedding, also called SNE, is a probabilistic approach and, by pairwise dissimilarity using Gaussian, is used to map the high-dimensional data distribution on the low-dimensional space,
     e.g. 2D space, for data visualization. Cost function, a sum of Kullback-Leibler divergences, is minimized using gradient descent.
    The aim of SNE is to map the high-dimensional data to low-dimensional space for data visualization.
    A good mapping approach is to obtain the distance between two mapped data points in low-dimensional space, which can reflect the distance between two data points in high-dimensional space.
    So, when people see the distance between 2 points in low-dimensional space, they can have idea of distance between 2 points in high-dimensional space, i.e. data visualization.
    For each object, i, and each potential neighbor, j, we start by computing the asymmetric probability pij, that i would pick j as its neighbor.
    The dissimilarities, dij², can be computed as the scaled squared Euclidean distance between two high-dimensional points, xi, xj where sigma_i is either set by hand or by binary searching.
    In the low-dimensional space, Gaussian neighborhoods are also used but with a fixed variance. The calculation of qij is similar to that of pij but qij is in low-dimensional space,
    and yi, yj, yk are those data points in low-dimensional space. The aim of the embedding/mapping is to match these two distributions as well as possible.
    This is achieved by minimizing a cost function which is a sum of Kullback-Leibler divergences between the original (pij) and induced (qij) distributions over neighbors for each object:
    Source: https://medium.com/swlh/review-sne-stochastic-neighbor-embedding-data-visualization-ef75880da6f7
    """
    def __init__(self, n_components, perplexity, lr=0.01, n_epochs=100):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_epochs = n_epochs

    def _compute_perplexity_from_sigma(self, data_matrix, center_idx, sigma):
        similarities = self._similarity(data_matrix[center_idx, :], data_matrix, sigma, "h")
        p = similarities / similarities.sum()
        shannon = - (p[p != 0] * torch.log2(p[p != 0])).sum()  # log(0) avoidance
        perp = 2 ** shannon.item()
        return perp

    def _search_sigmas(self, data_matrix):
        sigmas = torch.zeros(self.N)
        sigma_range = np.arange(0.1, 0.6, 0.1)
        for i in tqdm(range(self.N), desc="search sigma"):
            perps = np.zeros(len(sigma_range))
            for j, sigma in enumerate(sigma_range):
                perp = self._compute_perplexity_from_sigma(data_matrix, i, sigma)
                perps[j] = perp
            best_idx = (np.abs(perps - self.perplexity)).argmin()
            best_sigma = sigma_range[best_idx]
            sigmas[i] = best_sigma
        return sigmas

    def _similarity(self, x1, x2, sigma, mode):
        # SNE uses a normal distribution in both high and low dimensions
        return torch.exp(- ((x1 - x2) ** 2).sum(dim=1) / 2 * (sigma ** 2))

    def _compute_similarity(self, data_matrix, sigmas, mode):
        similarities = torch.zeros((self.N, self.N))
        for i in range(self.N):
            s_i = self._similarity(data_matrix[i, :], data_matrix, sigmas[i], mode)
            similarities[i] = s_i
        return similarities

    def _compute_cond_prob(self, similarities, mode):
        # In #SNE, the calculation of similarity does not change depending on the mode.
        cond_prob_matrix = torch.zeros((self.N, self.N))
        for i in range(self.N):
            p_i = similarities[i] / similarities[i].sum()
            cond_prob_matrix[i] = p_i
        return cond_prob_matrix

    def fit_transform(self, X, verbose=False):
        self.N = X.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.tensor(X, device=device)

        # 1. Randomly initialize y
        y = torch.randn(size=(self.N, self.n_components), requires_grad=True, device=device)
        optimizer = optim.Adam([y], lr=self.lr)

        # 2. Obtain the variance of the normal distribution corresponding to each data point in high-dimensional space from the specified perplexity.
        sigmas = self._search_sigmas(X)

        # 3. Find the similarity between each data point in high-dimensional space.
        X_similarities = self._compute_similarity(X, sigmas, "h")
        p = self._compute_cond_prob(X_similarities, "h")

        # 4. Repeat the following until it converges
        # TODO: this step runs on the full dataset, but possibly could be batched for faster convergence on large datasets
        loss_history = []
        for i in tqdm(range(self.n_epochs), desc="fitting"):
            optimizer.zero_grad()
            y_similarities = self._compute_similarity(y, torch.ones(self.N, device=device) / (2 ** (1/2)), "l")
            q = self._compute_cond_prob(y_similarities, "l")

            kl_loss = (p[p != 0] * (p[p != 0] / q[p != 0]).log()).mean()  # log(0) avoidance
            kl_loss.backward()
            loss_history.append(kl_loss.item())
            optimizer.step()

        if verbose:
            plt.plot(loss_history)
            plt.xlabel("epoch")
            plt.ylabel("loss")
        return y.cpu().detach().numpy()


class TSNE(SNE):
    """
    t-distributed stochastic neighbor embedding (t-SNE) is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map.
    It is based on Stochastic Neighbor Embedding originally developed by Sam Roweis and Geoffrey Hinton,[1] where Laurens van der Maaten proposed the t-distributed variant.
    It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. 
    Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects
    are modeled by distant points with high probability.

    The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects are assigned
    a higher probability while dissimilar points are assigned a lower probability.
    Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback–Leibler divergence (KL divergence) between the two
    distributions with respect to the locations of the points in the map. While the original algorithm uses the Euclidean distance between objects as the base of its similarity metric,
    this can be changed as appropriate.

    While t-SNE plots often seem to display clusters, the visual clusters can be influenced strongly by the chosen parameterization and therefore a good understanding of the parameters for t-SNE is necessary.
    Such "clusters" can be shown to even appear in non-clustered data,[11] and thus may be false findings.
    Interactive exploration may thus be necessary to choose parameters and validate results.
    It has been demonstrated that t-SNE is often able to recover well-separated clusters, and with special parameter choices, approximates a simple form of spectral clustering.
    Source: https://medium.com/swlh/review-sne-stochastic-neighbor-embedding-data-visualization-ef75880da6f7
    """
    def _similarity(self, x1, x2, sigma, mode):
        if mode == "h":
            return torch.exp(- ((x1 - x2) ** 2).sum(dim=1) / 2 * (sigma ** 2))
        if mode == "l":
            return (1 + ((x1 - x2) ** 2).sum(dim=1)) ** (-1)

    def _compute_cond_prob(self, similarities, mode):
        cond_prob_matrix = torch.zeros((self.N, self.N))
        for i in range(self.N):
            p_i = similarities[i] / similarities[i].mean()
            cond_prob_matrix[i] = p_i

        if mode == "h":
            cond_prob_matrix = (cond_prob_matrix + torch.t(cond_prob_matrix)) / 2
        return cond_prob_matrix
