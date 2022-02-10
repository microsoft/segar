from __future__ import annotations

__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
"""Generator from Inception model.

"""

import argparse
import glob
import os
import random
import requests
import shutil
from typing import Optional, Tuple, Type, Union
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from scipy.linalg import norm
import torch
from tqdm import tqdm

from segar import ASSET_DIR
from segar.factors import Factor
from segar.metrics import wasserstein_distance
from segar.rendering.generators.generator import Generator


INCEPTION_DIR = os.path.join(ASSET_DIR, "inception_linear")
INCEPTION_WEIGHTS = os.path.join(INCEPTION_DIR, "weights.npy")

_inception_paths = dict(
    baseline=os.path.join(INCEPTION_DIR, "baseline", "models"),
    close=os.path.join(INCEPTION_DIR, "close", "models"),
    far=os.path.join(INCEPTION_DIR, "far", "models"),
)


def generate_aligned_generative_models(
    n_models: int = 100,
    close_range: Tuple[float, float] = (1e-7, 0.25),
    far_range: Tuple[float, float] = (1.0, 100),
    out_path: str = None,
    model_name: str = "inception_linear",
):
    """Generates a bunch of generative models leveraging variation in
    kmeans solutions.

    These are aligned, that is the sets of features between models is the
    closest one can hope. This turns out to still generate meaningful
    variation in generative models.

    We also select subsets of models based on whether they are "close" or
    "far" based on their underlying features. This is used for
    generalization experiments.

    :param n_models: Number of models to generate.
    :param close_range: Range of distances where models are considered
        "close".
    :param far_range: Range of distances where models are considered "far".
    :param out_path: Output directory to save models and example patterns.
    :param model_name: Label used to save files.
    """
    models = []

    if model_name == "inception_linear":
        C = InceptionClusterGenerator

    for i in range(n_models):
        model = C(dim_x=16, dim_y=16, seed=i)
        if len(models) != 0:
            model.align_with(models[0])
        models.append(model)

    distances = []

    for model in models:
        distances.append(models[0].distance(model))

    s_idx = np.argsort(distances).tolist()

    def save(dir_path, idxs):
        if os.path.exists(f"{dir_path}/models/"):
            shutil.rmtree(f"{dir_path}/models/")
        if os.path.exists(f"{dir_path}/figures/"):
            shutil.rmtree(f"{dir_path}/figures/")

        os.makedirs(f"{dir_path}/models/")
        os.makedirs(f"{dir_path}/figures/")

        for i, idx in enumerate(idxs):
            model = models[idx]
            dist = distances[idx]
            p_model = f"{dir_path}/models/{model_name}_{i}_{dist:.2f}.npy"
            p_image = f"{dir_path}/figures/{model_name}_{i}_{dist:.2f}.png"

            model.save(p_model)
            model.show_patterns(p_image)

    if out_path is not None:
        baseline_path = os.path.join(out_path, "baseline")
        close_path = os.path.join(out_path, "close")
        far_path = os.path.join(out_path, "far")

        os.makedirs(baseline_path, exist_ok=True)
        os.makedirs(close_path, exist_ok=True)
        os.makedirs(far_path, exist_ok=True)

        baseline_idx = [i for i in s_idx if distances[i] < close_range[0]]

        close_idx = [
            i
            for i in s_idx
            if close_range[0] <= distances[i] <= close_range[1]
        ]

        far_idx = [
            i for i in s_idx if far_range[0] <= distances[i] <= far_range[1]
        ]

        save(baseline_path, baseline_idx)
        save(close_path, close_idx)
        save(far_path, far_idx)


class InceptionClusterGenerator(Generator):
    """Generates visual features according to a pretrained Inception model.

    We cluster the first layer weights using k means, assigning each cluster
    to a factor.

    """

    def __init__(
        self,
        grayscale: bool = False,
        n_clusters: int = None,
        dim_x: int = 64,
        dim_y: int = 64,
        stride: int = 1,
        n_rand_factors: int = 3,
        cluster_sampling: str = "sequential",
        model_path: Optional[str] = None,
        seed: int = 0,
    ):

        n_factors = len(self._factors) + n_rand_factors + 1
        super().__init__(
            dim_x,
            dim_y,
            n_factors,
            grayscale=grayscale,
            n_rand_factors=n_rand_factors,
            stride=stride,
            model_path=model_path,
        )
        n_clusters = n_clusters or self.n_factors

        feature_file = INCEPTION_WEIGHTS

        # Load the precomputed inception weights.
        if not os.path.exists(feature_file):
            self.download_weights()

        w = np.load(feature_file)
        if grayscale:
            w = w.mean(axis=1)

        # Remove the mean and variance of each feature.
        w_shape = w.shape
        w = w.reshape(w_shape[0], -1)
        w -= w.mean(axis=1, keepdims=True)
        w /= norm(w, axis=1, keepdims=True)

        self.features = w
        w = w.reshape(*w_shape)
        self.n_features = self.features.shape[0]
        self.cluster_sampling = cluster_sampling

        # Construct a transpose convolutional layer for generation using
        # weights.
        self.convt = torch.nn.ConvTranspose2d(
            self.n_features, self.n_channels, 3, stride, 1
        )
        self.convt.weight.data = torch.tensor(w)

        if self.model_path is None:
            # Cluster the weights using kmeans.
            data = self.features

            km = KMeans(n_clusters=n_clusters, random_state=seed).fit(data)
            self._km_labels = km.labels_
        else:
            self._km_labels = np.load(self.model_path)

        # TODO: This might cause an error if some labels aren't assigned.
        self.n_clusters = max(self._km_labels) + 1
        assert self.n_clusters >= self.n_factors

    @staticmethod
    def download_weights() -> None:
        """Downloads inception weights.
        Inception weights script taken from
            https://github.com/huyvnphan/PyTorch_CIFAR10.

        My assumption is that these are pretrained on Imagenet, as per how
        these weights are used for initialing models before training in the
        above codebase.
        """
        zip_path = os.path.join(INCEPTION_DIR, "state_dict.zip")
        model_path = os.path.join(INCEPTION_DIR, "inception_weights.pt")
        weights_path = INCEPTION_WEIGHTS

        # Begin copy from https://github.com/huyvnphan/PyTorch_CIFAR10/data.py

        if not os.path.exists(zip_path):
            url = (
                "https://rutgers.box.com/shared/static"
                "/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
            )

            # Streaming, so we can iterate over the response.
            r = requests.get(url, stream=True)

            # Total size in Mebibyte
            total_size = int(r.headers.get("content-length", 0))
            block_size = 2 ** 20  # Mebibyte
            t = tqdm(total=total_size, unit="MiB", unit_scale=True)

            with open(zip_path, "wb") as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()

            if total_size != 0 and t.n != total_size:
                raise Exception("Error, something went wrong")

        # End copy
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            source = zip_ref.open("state_dicts/inception_v3.pt")
            with open(model_path, "wb") as target:
                shutil.copyfileobj(source, target)

        model_weights = torch.load(model_path)
        weights = model_weights["Conv2d_1a_3x3.conv.weight"].detach().numpy()
        np.save(weights_path, weights)

    @classmethod
    def config_handler(cls, dist: str = "baseline") -> str:
        """Handler for passing configurations for Inception-generated visual
            features.

        :param dist: Which distribution to draw parameters from.
        :return: Dictionary of parameters to pass to constructor.
        """

        try:
            p = _inception_paths[dist]
        except KeyError:
            raise KeyError(
                f"Distribution {dist} not compatible "
                f"configuration with {cls.__name__}."
            )

        return p

    @property
    def parameters(self) -> np.ndarray:
        return self._km_labels

    def save(self, p_model: str):
        """Saves the model parameters.

        :param p_model: Path to save parameters.
        :return: None
        """
        np.save(p_model, self.parameters)

    def get_background(self, dim_x: int, dim_y: int) -> np.ndarray:
        """Generates a background image.

        This uses the first cluster only (reserved).

        :param dim_x: Number of pixels in the x dimension for generated
            visual features.
        :param dim_y: Number of pixels in the y dimension for generated
            visual features.
        :return: (np.array) Background image.
        """

        affordance_vec = torch.zeros(self.n_clusters)
        affordance_vec[0] = 1.0
        return self.gen_visual_features(
            affordance_vec, dim_x=dim_x, dim_y=dim_y
        )

    def gen_visual_features(
        self,
        factor_vec: torch.tensor,
        mag_scale: float = 1.0,
        dim_x: int = None,
        dim_y: int = None,
    ) -> Union[np.ndarray, None]:
        """Generates a visual feature according to an affordance vector.

        :param factor_vec: (torch.tensor) Vector of affordances.
        :param mag_scale: Scaling factor for the feature activations.
        :param dim_x: Number of pixels in the x dimension for generated
            visual features.
        :param dim_y: Number of pixels in the y dimension for generated
            visual features.
        :return: Pattern for this visual feature.
        """

        assert factor_vec.size(0) <= self.n_features

        dim_x, dim_y = self.get_rendering_dims(dim_x, dim_y)

        if sum(factor_vec ** 2) == 0:
            return

        acts = []
        for c, v in enumerate(factor_vec):
            if not v:
                continue
            # Which features correspond to which cluster.
            c_idxs = np.where(self._km_labels == c)[0]

            # Here we sample dim_x * dim_y features in the cluster
            # corresponding to the current factor (aka affordance here).
            if self.cluster_sampling == "random":
                # Randomly select cluster members.
                weights = torch.zeros(self.n_features)
                weights[c_idxs] = 1.0
                f_idxs = torch.multinomial(
                    weights, dim_x * dim_y, replacement=True
                )
            elif self.cluster_sampling == "sequential":
                # Randomly start at one of the cluster members, then sample
                # sequentially.
                start = random.randint(0, c_idxs.size)
                offsets = torch.arange(dim_x * dim_y) % c_idxs.size
                f_idxs = c_idxs[(start + offsets) % c_idxs.size]
            else:
                raise NotImplementedError(self.cluster_sampling)
            loc_acts = torch.zeros(dim_x * dim_y, self.n_features)
            adx = torch.arange(dim_x * dim_y).long()
            loc_acts[adx[:, None], f_idxs[:, None]] = v * mag_scale
            loc_acts = loc_acts.t().reshape(-1, dim_x, dim_y)[None, :, :, :]
            acts.append(loc_acts)

        # Generate.
        acts = torch.cat(acts, dim=0).sum(dim=0, keepdims=True)
        out = self.convt(acts)
        out = out[0].permute(1, 2, 0)
        out = 255.0 * (out - out.min()) / (out.max() - out.min())
        out = out.int().numpy().astype("uint8")
        return out

    def get_pattern(self, factor_dict: dict[Type[Factor], Factor]):
        """Returns the visual features for a set of attributes.

        """

        factor_list = [0.0]  # First index is background
        for factor_type in self._factors:
            factor = factor_dict.get(factor_type, 0.0)
            if isinstance(factor, Factor):
                factor = factor.value
            factor_list.append(factor)

        factor_array = np.array(factor_list)

        if self.n_rand_factors > 0:
            random_factors = np.clip(
                np.random.normal(0.0, 1.0, size=(self.n_rand_factors,)), -1, 1
            )
            factor_array = np.append(factor_array, random_factors)

        pattern = self.gen_visual_features(torch.tensor(factor_array).float())
        if pattern is not None:
            pattern = (
                np.zeros(
                    (self.dim_y, self.dim_x, self.n_channels), dtype=np.uint8
                )
                + pattern
            )
        return pattern

    def show_patterns(self, outpath: str = None):
        """Demos the patterns as a 2-d grid of factor combinations.

        :param outpath: Path to output the figure.
        :return: None
        """

        def image_grid(array, ncols=4):
            # From https://kanoki.org/2021/05/11/show-images-in-grid
            # -inside-jupyter-notebook-using-matplotlib-and-numpy/
            index, height, width, channels = array.shape
            nrows = index // ncols
            img_grid = (
                array.reshape(nrows, ncols, height, width, channels)
                .swapaxes(1, 2)
                .reshape(height * nrows, width * ncols, channels)
            )
            return img_grid

        imgs = []
        for i in range(self.n_factors):
            for j in range(self.n_factors):
                arr = torch.zeros(self.n_factors)
                arr[i] = 0.5
                arr[j] += 0.5
                imgs.append(self.gen_visual_features(arr))

        img_arr = image_grid(np.array(imgs), ncols=self.n_factors)
        plt.figure(figsize=(20.0, 20.0))
        plt.imshow(img_arr)
        if outpath is None:
            plt.show()
        else:
            plt.savefig(outpath)
        plt.close()

    def distance(self, other: InceptionClusterGenerator) -> float:
        """Calculates the wasserstein-2 distance between clusters.

        :param other: Other Inception-based clustering generator.
        :return: Distance between clusterings.
        """

        assert isinstance(other, InceptionClusterGenerator)
        assert self.n_clusters == other.n_clusters
        assert self.n_features == other.n_features

        d = 0.0
        for c in range(self.n_clusters):
            idx1 = np.where(self._km_labels == c)[0].tolist()
            idx2 = np.where(other._km_labels == c)[0].tolist()
            x1 = np.array(self.features[idx1])
            x2 = np.array(other.features[idx2])
            w = wasserstein_distance(x1, x2)

            d += w

        return d

    def align_with(self, other: InceptionClusterGenerator):
        """Aligns labels with other clustering output.

        Finds the best permutation of the labels such that assignments are
        closest to other's.

        :param other: Other Inception-based clustering generator.
        """
        assert isinstance(other, InceptionClusterGenerator)
        assert self.n_clusters == other.n_clusters
        assert self.n_features == other.n_features
        cost = -contingency_matrix(self._km_labels, other._km_labels)
        _, col_ind = linear_sum_assignment(cost)
        self._km_labels = np.array(
            [col_ind[i] for i in self._km_labels.tolist()]
        )

    @staticmethod
    def get_paths(dir_path: str) -> list[str]:
        model_paths = glob.glob(os.path.join(dir_path, "*.npy"))
        return model_paths


_MODELS = ("inception_linear",)


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Create some generative models for rendering the "
            "pixel observation space.",
        )

        parser.add_argument(
            "model",
            type=str,
            choices=_MODELS,
            help="Which generative model to use.",
        )

        parser.add_argument(
            "out_path",
            type=str,
            help="Path to directory where model parameters "
            "and figures will be saved.",
        )
        parser.add_argument(
            "--n_models",
            type=int,
            default=100,
            help="Number of models to generate.",
        )
        parser.add_argument(
            "--close_range",
            type=float,
            default=(1e-16, 0.5),
            metavar=("low", "high"),
            nargs=2,
            help="Range of distances where models are "
            "considered close to each other.",
        )
        parser.add_argument(
            "--far_range",
            type=float,
            default=(0.5, 100.0),
            metavar=("low", "high"),
            nargs=2,
            help="Range of distances where models are "
            "considered far to each other.",
        )

        return parser.parse_args()

    args = parse_args()

    generate_aligned_generative_models(
        n_models=args.n_models,
        out_path=args.out_path,
        close_range=args.close_range,
        far_range=args.far_range,
        model_name=args.model,
    )
