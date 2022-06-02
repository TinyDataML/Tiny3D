# Copyright (C) 2017-2022  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.


"""
Helper methods that are useful for benchmarking cleanlab’s core algorithms.
These methods introduce synthetic noise into the labels of a classification dataset.
Specifically, this module provides methods for generating valid noise matrices (for which learning with noise is possible),
generating noisy labels given a noise matrix, generating valid noise matrices with a specific trace value, and more.
"""

import numpy as np
from cleanlab.internal.util import value_counts
import warnings


def noise_matrix_is_valid(noise_matrix, py, *, verbose=False):
    """Given a prior `py` representing ``p(true_label=k)``, checks if the given `noise_matrix` is a
    learnable matrix. Learnability means that it is possible to achieve
    better than random performance, on average, for the amount of noise in
    `noise_matrix`.

    Parameters
    ----------
    noise_matrix : np.array
      An array of shape ``(K, K)`` representing the conditional probability
      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of
      examples in every class, labeled as every other class. Assumes columns of
      `noise_matrix` sum to 1.

    py : np.array
      An array of shape ``(K,)`` representing the fraction (prior probability)
      of each true class label, ``P(true_label = k)``.

    Returns
    -------
    bool
      Whether the noise matrix is a learnable matrix.
    """

    # Number of classes
    K = len(py)

    # let's assume some number of training examples for code readability,
    # but it doesn't matter what we choose as it's not actually used.
    N = float(10000)

    ps = np.dot(noise_matrix, py)  # P(true_label=k)

    # P(label=k, true_label=k')
    joint_noise = np.multiply(noise_matrix, py)  # / float(N)

    # Check that joint_probs is valid probability matrix
    if not (abs(joint_noise.sum() - 1.0) < 1e-6):
        return False

    # Check that noise_matrix is a valid matrix
    # i.e. check p(label=k)*p(true_label=k) < p(label=k, true_label=k)
    for i in range(K):
        C = N * joint_noise[i][i]
        E1 = N * joint_noise[i].sum() - C
        E2 = N * joint_noise.T[i].sum() - C
        O = N - E1 - E2 - C
        if verbose:
            print(
                "E1E2/C",
                round(E1 * E2 / C),
                "E1",
                round(E1),
                "E2",
                round(E2),
                "C",
                round(C),
                "|",
                round(E1 * E2 / C + E1 + E2 + C),
                "|",
                round(E1 * E2 / C),
                "<",
                round(O),
            )
            print(
                round(ps[i] * py[i]),
                "<",
                round(joint_noise[i][i]),
                ":",
                ps[i] * py[i] < joint_noise[i][i],
            )

        if not (ps[i] * py[i] < joint_noise[i][i]):
            return False

    return True


def generate_noisy_labels(true_labels, noise_matrix):
    """Generates noisy `labels` from perfect labels `true_labels`,
    "exactly" yielding the provided `noise_matrix` between `labels` and `true_labels`.

    Below we provide a for loop implementation of what this function does.
    We do not use this implementation as it is not a fast algorithm, but
    it explains as Python pseudocode what is happening in this function.

    Parameters
    ----------
    true_labels : np.array
      An array of shape ``(N,)`` representing perfect labels, without any
      noise. Contains K distinct natural number classes, 0, 1, ..., K-1.

    noise_matrix : np.array
      An array of shape ``(K, K)`` representing the conditional probability
      matrix ``P(label=k_s|true_label=k_y)`` containing the fraction of
      examples in every class, labeled as every other class. Assumes columns of
      `noise_matrix` sum to 1.

    Returns
    -------
    labels : np.array
      An array of shape ``(N,)`` of noisy labels.

    Examples
    --------

    .. code:: python

        # Generate labels
        count_joint = (noise_matrix * py * len(y)).round().astype(int)
        labels = np.array(y)
        for k_s in range(K):
            for k_y in range(K):
                if k_s != k_y:
                    idx_flip = np.where((labels==k_y)&(true_label==k_y))[0]
                    if len(idx_flip): # pragma: no cover
                        labels[np.random.choice(
                            idx_flip,
                            count_joint[k_s][k_y],
                            replace=False,
                        )] = k_s
    """

    # Make y a numpy array, if it is not
    true_labels = np.asarray(true_labels)

    # Number of classes
    K = len(noise_matrix)

    # Compute p(true_label=k)
    py = value_counts(true_labels) / float(len(true_labels))

    # Counts of pairs (labels, y)
    count_joint = (noise_matrix * py * len(true_labels)).astype(int)
    # Remove diagonal entries as they do not involve flipping of labels.
    np.fill_diagonal(count_joint, 0)

    # Generate labels
    labels = np.array(true_labels)
    for k in range(K):  # Iterate over true_label == k
        # Get the noisy labels that have non-zero counts
        labels_per_class = np.where(count_joint[:, k] != 0)[0]
        # Find out how many of each noisy  label we need to flip to
        label_counts = count_joint[labels_per_class, k]
        # Create a list of the new noisy labels
        noise = [labels_per_class[i] for i, c in enumerate(label_counts) for z in range(c)]
        # Randomly choose y labels for class k and set them to the noisy labels.
        idx_flip = np.where((labels == k) & (true_labels == k))[0]
        if len(idx_flip) and len(noise) and len(idx_flip) >= len(noise):  # pragma: no cover
            labels[np.random.choice(idx_flip, len(noise), replace=False)] = noise

    # Validate that labels indeed produces the correct noise_matrix (or close to it)
    # Compute the actual noise matrix induced by labels
    # counts = confusion_matrix(labels, true_labels).astype(float)
    # new_noise_matrix = counts / counts.sum(axis=0)
    # assert(np.linalg.norm(noise_matrix - new_noise_matrix) <= 2)

    return labels


def generate_noise_matrix_from_trace(
    K,
    trace,
    *,
    max_trace_prob=1.0,
    min_trace_prob=1e-5,
    max_noise_rate=1 - 1e-5,
    min_noise_rate=0.0,
    valid_noise_matrix=True,
    py=None,
    frac_zero_noise_rates=0.0,
    seed=0,
    max_iter=10000,
):
    """Generates a ``K x K`` noise matrix ``P(label=k_s|true_label=k_y)`` with
    ``np.sum(np.diagonal(noise_matrix))`` equal to the given `trace`.

    Parameters
    ----------
    K : int
      Creates a noise matrix of shape ``(K, K)``. Implies there are
      K classes for learning with noisy labels.

    trace : float
      Sum of diagonal entries of array of random probabilities returned.

    max_trace_prob : float
      Maximum probability of any entry in the trace of the return matrix.

    min_trace_prob : float
      Minimum probability of any entry in the trace of the return matrix.

    max_noise_rate : float
      Maximum noise_rate (non-diagonal entry) in the returned np.array.

    min_noise_rate : float
      Minimum noise_rate (non-diagonal entry) in the returned np.array.

    valid_noise_matrix : bool, default=True
      If ``True``, returns a matrix having all necessary conditions for
      learning with noisy labels. In particular, ``p(true_label=k)p(label=k) < p(true_label=k,label=k)``
      is satisfied. This requires that ``trace > 1``.

    py : np.array
      An array of shape ``(K,)`` representing the fraction (prior probability) of each true class label, ``P(true_label = k)``.
      This argument is **required** when ``valid_noise_matrix=True``.

    frac_zero_noise_rates : float
      The fraction of the ``n*(n-1)`` noise rates
      that will be set to 0. Note that if you set a high trace, it may be
      impossible to also have a low fraction of zero noise rates without
      forcing all non-1 diagonal values. Instead, when this happens we only
      guarantee to produce a noise matrix with `frac_zero_noise_rates` *or
      higher*. The opposite occurs with a small trace.

    seed : int
      Seeds the random number generator for numpy.

    max_iter : int, default=10000
      The max number of tries to produce a valid matrix before returning ``None``.

    Returns
    -------
    noise_matrix : np.array or None
      An array of shape ``(K, K)`` representing the noise matrix ``P(label=k_s|true_label=k_y)`` with `trace`
      equal to ``np.sum(np.diagonal(noise_matrix))``. This a conditional probability matrix and a
      left stochastic matrix. Returns ``None`` if `max_iter` is exceeded.
    """

    if valid_noise_matrix and trace <= 1:
        raise ValueError(
            "trace = {}. trace > 1 is necessary for a".format(trace)
            + " valid noise matrix to be returned (valid_noise_matrix == True)"
        )

    if valid_noise_matrix and py is None and K > 2:
        raise ValueError(
            "py must be provided (not None) if the input parameter" + " valid_noise_matrix == True"
        )

    if K <= 1:
        raise ValueError("K must be >= 2, but K = {}.".format(K))

    if max_iter < 1:
        return None

    np.random.seed(seed)

    # Special (highly constrained) case with faster solution.
    # Every 2 x 2 noise matrix with trace > 1 is valid because p(y) is not used
    if K == 2:
        if frac_zero_noise_rates >= 0.5:  # Include a single zero noise rate
            noise_mat = np.array(
                [
                    [1.0, 1 - (trace - 1.0)],
                    [0.0, trace - 1.0],
                ]
            )
            return noise_mat if np.random.rand() > 0.5 else np.rot90(noise_mat, k=2)
        else:  # No zero noise rates
            diag = generate_n_rand_probabilities_that_sum_to_m(2, trace)
            noise_matrix = np.array(
                [
                    [diag[0], 1 - diag[1]],
                    [1 - diag[0], diag[1]],
                ]
            )
            return noise_matrix

            # K > 2
    for z in range(max_iter):
        noise_matrix = np.zeros(shape=(K, K))

        # Randomly generate noise_matrix diagonal.
        nm_diagonal = generate_n_rand_probabilities_that_sum_to_m(
            n=K,
            m=trace,
            max_prob=max_trace_prob,
            min_prob=min_trace_prob,
        )
        np.fill_diagonal(noise_matrix, nm_diagonal)

        # Randomly distribute number of zero-noise-rates across columns
        num_col_with_noise = K - np.count_nonzero(1 == nm_diagonal)
        num_zero_noise_rates = int(K * (K - 1) * frac_zero_noise_rates)
        # Remove zeros already in [1,0,..,0] columns
        num_zero_noise_rates -= (K - num_col_with_noise) * (K - 1)
        num_zero_noise_rates = np.maximum(num_zero_noise_rates, 0)  # Prevent negative
        num_zero_noise_rates_per_col = (
            randomly_distribute_N_balls_into_K_bins(
                N=num_zero_noise_rates,
                K=num_col_with_noise,
                max_balls_per_bin=K - 2,
                # 2 = one for diagonal, and one to sum to 1
                min_balls_per_bin=0,
            )
            if K > 2
            else np.array([0, 0])
        )  # Special case when K == 2
        stack_nonzero_noise_rates_per_col = list(K - 1 - num_zero_noise_rates_per_col)[::-1]
        # Randomly generate noise rates for columns with noise.
        for col in np.arange(K)[nm_diagonal != 1]:
            num_noise = stack_nonzero_noise_rates_per_col.pop()
            # Generate num_noise noise_rates for the given column.
            noise_rates_col = list(
                generate_n_rand_probabilities_that_sum_to_m(
                    n=num_noise,
                    m=1 - nm_diagonal[col],
                    max_prob=max_noise_rate,
                    min_prob=min_noise_rate,
                )
            )
            # Randomly select which rows of the noisy column to assign the
            # random noise rates
            rows = np.random.choice(
                [row for row in range(K) if row != col], num_noise, replace=False
            )
            for row in rows:
                noise_matrix[row][col] = noise_rates_col.pop()
        if not valid_noise_matrix or noise_matrix_is_valid(noise_matrix, py):
            return noise_matrix

    return None


def generate_n_rand_probabilities_that_sum_to_m(
    n,
    m,
    *,
    max_prob=1.0,
    min_prob=0.0,
):
    """
    Generates `n` random probabilities that sum to `m`.

    When ``min_prob=0`` and ``max_prob = 1.0``, use
    ``np.random.dirichlet(np.ones(n))*m`` instead.

    Parameters
    ----------
    n : int
      Length of array of random probabilities to be returned.

    m : float
      Sum of array of random probabilities that is returned.

    max_prob : float, default=1.0
      Maximum probability of any entry in the returned array. Must be between 0 and 1.

    min_prob : float, default=0.0
      Minimum probability of any entry in the returned array. Must be between 0 and 1.
    """

    epsilon = 1e-6  # Imprecision allowed for inequalities with floats

    if n == 0:
        return np.array([])
    if (max_prob + epsilon) < m / float(n):
        raise ValueError(
            "max_prob must be greater or equal to m / n, but "
            + "max_prob = "
            + str(max_prob)
            + ", m = "
            + str(m)
            + ", n = "
            + str(n)
            + ", m / n = "
            + str(m / float(n))
        )
    if min_prob > (m + epsilon) / float(n):
        raise ValueError(
            "min_prob must be less or equal to m / n, but "
            + "max_prob = "
            + str(max_prob)
            + ", m = "
            + str(m)
            + ", n = "
            + str(n)
            + ", m / n = "
            + str(m / float(n))
        )

    # When max_prob = 1, min_prob = 0, the next two lines are equivalent to:
    #   intermediate = np.sort(np.append(np.random.uniform(0, 1, n-1), [0, 1]))
    #   result = (intermediate[1:] - intermediate[:-1]) * m
    result = np.random.dirichlet(np.ones(n)) * m

    min_val = min(result)
    max_val = max(result)
    while max_val > (max_prob + epsilon):
        new_min = min_val + (max_val - max_prob)
        # This adjustment prevents the new max from always being max_prob.
        adjustment = (max_prob - new_min) * np.random.rand()
        result[np.argmin(result)] = new_min + adjustment
        result[np.argmax(result)] = max_prob - adjustment
        min_val = min(result)
        max_val = max(result)

    min_val = min(result)
    max_val = max(result)
    while min_val < (min_prob - epsilon):
        min_val = min(result)
        max_val = max(result)
        new_max = max_val - (min_prob - min_val)
        # This adjustment prevents the new min from always being min_prob.
        adjustment = (new_max - min_prob) * np.random.rand()
        result[np.argmax(result)] = new_max - adjustment
        result[np.argmin(result)] = min_prob + adjustment
        min_val = min(result)
        max_val = max(result)

    return result


def randomly_distribute_N_balls_into_K_bins(
    N,  # int
    K,  # int
    *,
    max_balls_per_bin=None,
    min_balls_per_bin=None,
):
    """Returns a uniformly random numpy integer array of length N that sums
    to K.

    Parameters
    ----------
    N : int
      Number of balls.
    K : int
      Number of bins.
    max_balls_per_bin : int
      Ensure that each bin contains at most `max_balls_per_bin` balls.
    min_balls_per_bin : int
      Ensure that each bin contains at least `min_balls_per_bin` balls.

    Returns
    -------
    np.array
    """

    if N == 0:
        return np.zeros(K, dtype=int)
    if max_balls_per_bin is None:
        max_balls_per_bin = N
    else:
        max_balls_per_bin = min(max_balls_per_bin, N)
    if min_balls_per_bin is None:
        min_balls_per_bin = 0
    else:
        min_balls_per_bin = min(min_balls_per_bin, N / K)
    if N / float(K) > max_balls_per_bin:
        N = max_balls_per_bin * K

    arr = np.round(
        generate_n_rand_probabilities_that_sum_to_m(
            n=K,
            m=1,
            max_prob=max_balls_per_bin / float(N),
            min_prob=min_balls_per_bin / float(N),
        )
        * N
    )
    while sum(arr) != N:
        while sum(arr) > N:  # pragma: no cover
            arr[np.argmax(arr)] -= 1
        while sum(arr) < N:
            arr[np.argmin(arr)] += 1
    return arr.astype(int)
