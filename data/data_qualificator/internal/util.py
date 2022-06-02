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


# ## Confident Learning Utilities
#
# #### Contains ancillary helper functions used throughout this package.

import numpy as np
from sklearn.utils import check_X_y


def assert_inputs_are_valid(X, s, pred_probs=None, allow_empty_X=False):  # pragma: no cover
    """Checks that X, labels, and pred_probs are correctly formatted"""

    if pred_probs is not None:
        if not isinstance(pred_probs, (np.ndarray, np.generic)):
            raise TypeError("pred_probs should be a numpy array.")
        if len(pred_probs) != len(s):
            raise ValueError("pred_probs and labels must have same length.")
        # Check for valid probabilities.
        if (pred_probs < 0).any() or (pred_probs > 1).any():
            raise ValueError("Values in pred_probs must be between 0 and 1.")

    if not isinstance(s, (np.ndarray, np.generic)):
        raise TypeError("labels should be a numpy array.")

    # Check that labels is zero-indexed (first label is 0).
    unique_classes = np.unique(s)
    if all(unique_classes != np.arange(len(unique_classes))):
        msg = "cleanlab requires zero-indexed labels (0,1,2,..,m-1), but in "
        msg += "your case: np.unique(labels) = {}".format(str(unique_classes))
        raise TypeError(msg)

    if not allow_empty_X:
        if X is None:
            raise ValueError("X cannot be None.")

        check_X_y(
            X,
            s,
            accept_sparse=True,
            dtype=None,
            force_all_finite=False,
            ensure_2d=False,
            allow_nd=True,
        )


def remove_noise_from_class(noise_matrix, class_without_noise):
    """A helper function in the setting of PU learning.
    Sets all P(label=class_without_noise|true_label=any_other_class) = 0
    in noise_matrix for pulearning setting, where we have
    generalized the positive class in PU learning to be any
    class of choosing, denoted by class_without_noise.

    Parameters
    ----------
    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix of the form P(label=k_s|true_label=k_y) containing
        the fraction of examples in every class, labeled as every other class.
        Assumes columns of noise_matrix sum to 1.

    class_without_noise : int
        Integer value of the class that has no noise. Traditionally,
        this is 1 (positive) for PU learning."""

    # Number of classes
    K = len(noise_matrix)

    cwn = class_without_noise
    x = np.copy(noise_matrix)

    # Set P( labels = cwn | y != cwn) = 0 (no noise)
    x[cwn, [i for i in range(K) if i != cwn]] = 0.0

    # Normalize columns by increasing diagonal terms
    # Ensures noise_matrix is a valid probability matrix
    for i in range(K):
        x[i][i] = 1 - float(np.sum(x[:, i]) - x[i][i])

    return x


def clip_noise_rates(noise_matrix):
    """Clip all noise rates to proper range [0,1), but
    do not modify the diagonal terms because they are not
    noise rates.

    ASSUMES noise_matrix columns sum to 1.

    Parameters
    ----------
    noise_matrix : np.array of shape (K, K), K = number of classes
        A conditional probability matrix containing the fraction of
        examples in every class, labeled as every other class.
        Diagonal terms are not noise rates, but are consistency P(label=k|true_label=k)
        Assumes columns of noise_matrix sum to 1"""

    def clip_noise_rate_range(noise_rate):
        """Clip noise rate P(label=k'|true_label=k) or P(true_label=k|label=k')
        into proper range [0,1)"""
        return min(max(noise_rate, 0.0), 0.9999)

    # Vectorize clip_noise_rate_range for efficiency with np.arrays.
    vectorized_clip = np.vectorize(clip_noise_rate_range)

    # Preserve because diagonal entries are not noise rates.
    diagonal = np.diagonal(noise_matrix)

    # Clip all noise rates (efficiently).
    noise_matrix = vectorized_clip(noise_matrix)

    # Put unmodified diagonal back.
    np.fill_diagonal(noise_matrix, diagonal)

    # Re-normalized noise_matrix so that columns sum to one.
    noise_matrix = noise_matrix / noise_matrix.sum(axis=0)

    return noise_matrix


def clip_values(x, low=0.0, high=1.0, new_sum=None):
    """Clip all values in p to range [low,high].
    Preserves sum of x.

    Parameters
    ----------
    x : np.array
        An array / list of values to be clipped.

    low : float
        values in x greater than 'low' are clipped to this value

    high : float
        values in x greater than 'high' are clipped to this value

    new_sum : float
        normalizes x after clipping to sum to new_sum

    Returns
    -------
    x : np.array
        A list of clipped values, summing to the same sum as x."""

    def clip_range(a, low=low, high=high):
        """Clip a into range [low,high]"""
        return min(max(a, low), high)

    # Vectorize clip_range for efficiency with np.arrays.
    vectorized_clip = np.vectorize(clip_range)

    # Store previous sum
    prev_sum = sum(x) if new_sum is None else new_sum

    # Clip all values (efficiently).
    x = vectorized_clip(x)

    # Re-normalized values to sum to previous sum.
    x = x * prev_sum / float(sum(x))

    return x


def value_counts(x):
    """Returns an np.array of shape (K, 1), with the
    value counts for every unique item in the labels list/array,
    where K is the number of unique entries in labels.

    Why this matters? Here is an example:

    .. code:: python

        x = [np.random.randint(0,100) for i in range(100000)]

    .. code:: ipython3

        %timeit np.bincount(x)
        # Result: 100 loops, best of 3: 3.9 ms per loop

    .. code:: ipython3

        %timeit np.unique(x, return_counts=True)[1]
        # Result: 100 loops, best of 3: 7.47 ms per loop

    Parameters
    ----------
    x : list or np.array (one dimensional)
        A list of discrete objects, like lists or strings, for
        example, class labels 'y' when training a classifier.
        e.g. ["dog","dog","cat"] or [1,2,0,1,1,0,2]"""
    try:
        return x.value_counts()
    except:
        if type(x[0]) is int and (np.array(x) >= 0).all():
            return np.bincount(x)
        else:
            return np.unique(x, return_counts=True)[1]


def round_preserving_sum(iterable):
    """Rounds an iterable of floats while retaining the original summed value.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    The while loop in this code was adapted from:
    https://github.com/cgdeboer/iteround

    Parameters
    -----------
    iterable : list<float> or np.array<float>
        An iterable of floats

    Returns
    -------
    list<int> or np.array<int>
        The iterable rounded to int, preserving sum."""

    floats = np.asarray(iterable, dtype=float)
    ints = floats.round()
    orig_sum = np.sum(floats).round()
    int_sum = np.sum(ints).round()
    # Adjust the integers so that they sum to orig_sum
    while abs(int_sum - orig_sum) > 1e-6:
        diff = np.round(orig_sum - int_sum)
        increment = -1 if int(diff < 0.0) else 1
        changes = min(int(abs(diff)), len(iterable))
        # Orders indices by difference. Increments # of changes.
        indices = np.argsort(floats - ints)[::-increment][:changes]
        for i in indices:
            ints[i] = ints[i] + increment
        int_sum = np.sum(ints).round()
    return ints.astype(int)


def round_preserving_row_totals(confident_joint):
    """Rounds confident_joint cj to type int
    while preserving the totals of reach row.
    Assumes that cj is a 2D np.array of type float.

    Parameters
    ----------
    confident_joint : 2D np.array<float> of shape (K, K)
        See compute_confident_joint docstring for details.

    Returns
    -------
    confident_joint : 2D np.array<int> of shape (K,K)
        Rounded to int while preserving row totals."""

    return np.apply_along_axis(
        func1d=round_preserving_sum,
        axis=1,
        arr=confident_joint,
    ).astype(int)


def int2onehot(labels):
    """Convert list of lists to a onehot matrix for multi-labels

    Parameters
    ----------
    labels: list of lists of integers
      e.g. [[0,1], [3], [1,2,3], [1], [2]]
      All integers from 0,1,...,K-1 must be represented."""

    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(labels)


def onehot2int(onehot_matrix):
    """Convert a onehot matrix for multi-labels to a list of lists of ints

    Parameters
    ----------
    onehot_matrix: 2D np.array of 0s and 1s
      A one hot encoded matrix representation of multi-labels.

    Returns
    -------
    labels: list of lists of integers
      e.g. [[0,1], [3], [1,2,3], [1], [2]]
      All integers from 0,1,...,K-1 must be represented."""

    return [list(np.where(row == 1)[0]) for row in onehot_matrix]


def estimate_pu_f1(s, prob_s_eq_1):
    """Computes Claesen's estimate of f1 in the pulearning setting.

    Parameters
    ----------
    s : iterable (list or np.array)
      Binary label (whether each element is labeled or not) in pu learning.

    prob_s_eq_1 : iterable (list or np.array)
      The probability, for each example, whether it has label=1 P(label=1|x)

    Output (float)
    ------
    Claesen's estimate for f1 in the pulearning setting."""

    pred = np.asarray(prob_s_eq_1) >= 0.5
    true_positives = sum((np.asarray(s) == 1) & (np.asarray(pred) == 1))
    all_positives = sum(s)
    recall = true_positives / float(all_positives)
    frac_positive = sum(pred) / float(len(s))
    return recall**2 / (2.0 * frac_positive) if frac_positive != 0 else np.nan


def confusion_matrix(true, pred):
    """Implements a confusion matrix for true labels
    and predicted labels. true and pred MUST BE the same length
    and have the same distinct set of class labels represented.

    Results are identical (and similar computation time) to:
        "sklearn.metrics.confusion_matrix"

    However, this function avoids the dependency on sklearn.

    Parameters
    ----------
    true : np.array 1d
      Contains labels.
      Assumes true and pred contains the same set of distinct labels.

    pred : np.array 1d
      A discrete vector of noisy labels, i.e. some labels may be erroneous.
      *Format requirements*: for dataset with K classes, labels must be in {0,1,...,K-1}.

    Returns
    -------
    confusion_matrix : np.array (2D)
      matrix of confusion counts with true on rows and pred on columns."""

    assert len(true) == len(pred)
    true_classes = np.unique(true)
    pred_classes = np.unique(pred)
    K_true = len(true_classes)  # Number of classes in true
    K_pred = len(pred_classes)  # Number of classes in pred
    map_true = dict(zip(true_classes, range(K_true)))
    map_pred = dict(zip(pred_classes, range(K_pred)))

    result = np.zeros((K_true, K_pred))
    for i in range(len(true)):
        result[map_true[true[i]]][map_pred[pred[i]]] += 1

    return result


def print_square_matrix(
    matrix,
    left_name="s",
    top_name="y",
    title=" A square matrix",
    short_title="s,y",
    round_places=2,
):
    """Pretty prints a matrix.

    Parameters
    ----------
    matrix : np.array
        the matrix to be printed
    left_name : str
        the name of the variable on the left of the matrix
    top_name : str
        the name of the variable on the top of the matrix
    title : str
        Prints this string above the printed square matrix.
    short_title : str
        A short title (6 characters or fewer) like P(labels|y) or P(labels,y).
    round_places : int
        Number of decimals to show for each matrix value."""

    short_title = short_title[:6]
    K = len(matrix)  # Number of classes
    # Make sure matrix is 2d array
    if len(np.shape(matrix)) == 1:
        matrix = np.array([matrix])
    print()
    print(title, "of shape", matrix.shape)
    print(" " + short_title + "".join(["\t" + top_name + "=" + str(i) for i in range(K)]))
    print("\t---" * K)
    for i in range(K):
        entry = "\t".join([str(z) for z in list(matrix.round(round_places)[i, :])])
        print(left_name + "=" + str(i) + " |\t" + entry)
    print("\tTrace(matrix) =", np.round(np.trace(matrix), round_places))
    print()


def print_noise_matrix(noise_matrix, round_places=2):
    """Pretty prints the noise matrix."""
    print_square_matrix(
        noise_matrix,
        title=" Noise Matrix (aka Noisy Channel) P(given_label|true_label)",
        short_title="p(s|y)",
        round_places=round_places,
    )


def print_inverse_noise_matrix(inverse_noise_matrix, round_places=2):
    """Pretty prints the inverse noise matrix."""
    print_square_matrix(
        inverse_noise_matrix,
        left_name="y",
        top_name="s",
        title=" Inverse Noise Matrix P(true_label|given_label)",
        short_title="p(y|s)",
        round_places=round_places,
    )


def print_joint_matrix(joint_matrix, round_places=2):
    """Pretty prints the joint label noise matrix."""
    print_square_matrix(
        joint_matrix,
        title=" Joint Label Noise Distribution Matrix P(given_label, true_label)",
        short_title="p(s,y)",
        round_places=round_places,
    )


def compress_int_array(int_array, num_possible_values):
    """Compresses dtype of np.array<int> if num_possible_values is small enough."""
    compressed_type = None
    if num_possible_values < np.iinfo(np.dtype("int16")).max:
        compressed_type = "int16"
    elif num_possible_values < np.iinfo(np.dtype("int32")).max:  # pragma: no cover
        compressed_type = "int32"  # pragma: no cover
    if compressed_type is not None:
        int_array = int_array.astype(compressed_type)
    return int_array


def smart_display_dataframe(df):  # pragma: no cover
    """Display a pandas dataframe if in a jupyter notebook, otherwise print it to console."""
    try:
        from IPython.display import display

        display(df)
    except:
        print(df)
