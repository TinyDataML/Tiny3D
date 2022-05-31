
def find_label_issues(
    self,
    X=None,
    labels=None,
    *,
    pred_probs=None,
    thresholds=None,
    noise_matrix=None,
    inverse_noise_matrix=None,
    save_space=False,
    clf_kwargs={},
):
    """
    Identifies potential label issues in the dataset using confident learning.

    Runs cross-validation to get out-of-sample pred_probs from `clf`
    and then calls :py:func:`filter.find_label_issues
    <cleanlab.filter.find_label_issues>` to find label issues.
    These label issues are cached internally and returned in a pandas DataFrame.
    Kwargs for :py:func:`filter.find_label_issues
    <cleanlab.filter.find_label_issues>` must have already been specified
    in the initialization of this class, not here.

    Unlike :py:func:`filter.find_label_issues
    <cleanlab.filter.find_label_issues>`, which requires `pred_probs`,
    this method only requires a classifier and it can do the cross-validation for you.
    Both methods return the same boolean mask that identifies which examples have label issues.
    This is the preferred method to use if you plan to subsequently invoke:
    :py:meth:`CleanLearning.fit()
    <cleanlab.classification.CleanLearning.fit>`.

    Note: this method computes the label issues from scratch. To access
    previously-computed label issues from this :py:class:`CleanLearning
    <cleanlab.classification.CleanLearning>` instance, use the
    :py:meth:`get_label_issues
    <cleanlab.classification.CleanLearning.get_label_issues>` method.

    This is the method called to find label issues inside
    :py:meth:`CleanLearning.fit()
    <cleanlab.classification.CleanLearning.fit>`
    and they share mostly the same parameters.

    Parameters
    ----------
    save_space : bool, optional
        If True, then returned `label_issues_df` will not be stored as attribute.
        This means some other methods like `self.get_label_issues()` will no longer work.


    For info about the **other parameters**, see the docstring of :py:meth:`CleanLearning.fit()
    <cleanlab.classification.CleanLearning.fit>`.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame of label issues for each example.
        Unless `save_space` argument is specified, same DataFrame is also stored as
        `self.label_issues_df` attribute accessible via
        :py:meth:`get_label_issues<cleanlab.classification.CleanLearning.get_label_issues>`.
        Each row represents an example from our dataset and
        the DataFrame may contain the following columns:

        * *is_label_issue*: boolean mask for the entire dataset where ``True`` represents a label issue and ``False`` represents an example that is accurately labeled with high confidence. This column is equivalent to `label_issues_mask` output from :py:func:`filter.find_label_issues<cleanlab.filter.find_label_issues>`.
        * *label_quality*: Numeric score that measures the quality of each label (how likely it is to be correct, with lower scores indicating potentially erroneous labels).
        * *given_label*: Integer indices corresponding to the class label originally given for this example (same as `labels` input). Included here for ease of comparison against `clf` predictions, only present if "predicted_label" column is present.
        * *predicted_label*: Integer indices corresponding to the class predicted by trained `clf` model. Only present if ``pred_probs`` were provided as input or computed during label-issue-finding.
        * *sample_weight*: Numeric values used to weight examples during the final training of `clf` in :py:meth:`CleanLearning.fit()<cleanlab.classification.CleanLearning.fit>`. This column not be present after `self.find_label_issues()` but may be added after call to :py:meth:`CleanLearning.fit()<cleanlab.classification.CleanLearning.fit>`. For more precise definition of sample weights, see documentation of :py:meth:`CleanLearning.fit()<cleanlab.classification.CleanLearning.fit>`
    """

    # Check inputs
    allow_empty_X = False if pred_probs is None else True
    assert_inputs_are_valid(X, labels, pred_probs, allow_empty_X=allow_empty_X)
    if noise_matrix is not None and np.trace(noise_matrix) <= 1:
        t = np.round(np.trace(noise_matrix), 2)
        raise ValueError("Trace(noise_matrix) is {}, but must exceed 1.".format(t))
    if inverse_noise_matrix is not None and (np.trace(inverse_noise_matrix) <= 1):
        t = np.round(np.trace(inverse_noise_matrix), 2)
        raise ValueError("Trace(inverse_noise_matrix) is {}. Must exceed 1.".format(t))

    # Number of classes
    self.num_classes = len(np.unique(labels))
    if len(labels) / self.num_classes < self.cv_n_folds:
        raise ValueError(
            "Need more data from each class for cross-validation. "
            "Try decreasing cv_n_folds (eg. to 2 or 3) in CleanLearning()"
        )
    # 'ps' is p(labels=k)
    self.ps = value_counts(labels) / float(len(labels))

    self.clf_kwargs = clf_kwargs
    self._process_label_issues_kwargs(self.find_label_issues_kwargs)
    # self._process_label_issues_kwargs might set self.confident_joint. If so, we should use it.
    if self.confident_joint is not None:
        self.py, noise_matrix, inv_noise_matrix = estimate_latent(
            confident_joint=self.confident_joint,
            labels=labels,
        )

    # If needed, compute noise rates (probability of class-conditional mislabeling).
    if noise_matrix is not None:
        self.noise_matrix = noise_matrix
        if inverse_noise_matrix is None:
            if self.verbose:
                print("Computing label noise estimates from provided noise matrix ...")
            self.py, self.inverse_noise_matrix = compute_py_inv_noise_matrix(
                ps=self.ps,
                noise_matrix=self.noise_matrix,
            )
    if inverse_noise_matrix is not None:
        self.inverse_noise_matrix = inverse_noise_matrix
        if noise_matrix is None:
            if self.verbose:
                print("Computing label noise estimates from provided inverse noise matrix ...")
            self.noise_matrix = compute_noise_matrix_from_inverse(
                ps=self.ps,
                inverse_noise_matrix=self.inverse_noise_matrix,
            )

    if noise_matrix is None and inverse_noise_matrix is None:
        if pred_probs is None:
            if self.verbose:
                print(
                    "Computing out of sample predicted probabilities via "
                    f"{self.cv_n_folds}-fold cross validation. May take a while ..."
                )
            (
                self.py,
                self.noise_matrix,
                self.inverse_noise_matrix,
                self.confident_joint,
                pred_probs,
            ) = estimate_py_noise_matrices_and_cv_pred_proba(
                X=X,
                labels=labels,
                clf=self.clf,
                cv_n_folds=self.cv_n_folds,
                thresholds=thresholds,
                converge_latent_estimates=self.converge_latent_estimates,
                seed=self.seed,
                clf_kwargs=self.clf_kwargs,
            )
        else:  # pred_probs is provided by user (assumed holdout probabilities)
            if self.verbose:
                print("Computing label noise estimates from provided pred_probs ...")
            (
                self.py,
                self.noise_matrix,
                self.inverse_noise_matrix,
                self.confident_joint,
            ) = estimate_py_and_noise_matrices_from_probabilities(
                labels=labels,
                pred_probs=pred_probs,
                thresholds=thresholds,
                converge_latent_estimates=self.converge_latent_estimates,
            )
    # If needed, compute P(label=k|x), denoted pred_probs (the predicted probabilities)
    if pred_probs is None:
        if self.verbose:
            print(
                "Computing out of sample predicted probabilities via "
                f"{self.cv_n_folds}-fold cross validation. May take a while ..."
            )

        pred_probs = estimate_cv_predicted_probabilities(
            X=X,
            labels=labels,
            clf=self.clf,
            cv_n_folds=self.cv_n_folds,
            seed=self.seed,
            clf_kwargs=self.clf_kwargs,
        )
    # If needed, compute the confident_joint (e.g. occurs if noise_matrix was given)
    if self.confident_joint is None:
        self.confident_joint = compute_confident_joint(
            labels=labels,
            pred_probs=pred_probs,
            thresholds=thresholds,
        )
    # if pulearning == the integer specifying the class without noise.
    if self.num_classes == 2 and self.pulearning is not None:  # pragma: no cover
        # pulearning = 1 (no error in 1 class) implies p(label=1|true_label=0) = 0
        self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
        self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
        # pulearning = 1 (no error in 1 class) implies p(true_label=0|label=1) = 0
        self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
        self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
        # pulearning = 1 (no error in 1 class) implies p(label=1,true_label=0) = 0
        self.confident_joint[self.pulearning][1 - self.pulearning] = 0
        self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

    if self.verbose:
        print("Using predicted probabilities to identify label issues ...")
    label_issues_mask = filter.find_label_issues(
        labels,
        pred_probs,
        **self.find_label_issues_kwargs,
    )
    label_quality_scores = get_label_quality_scores(
        labels, pred_probs, **self.label_quality_scores_kwargs
    )
    label_issues_df = pd.DataFrame(
        {"is_label_issue": label_issues_mask, "label_quality": label_quality_scores}
    )
    if self.verbose:
        print(f"Identified {np.sum(label_issues_mask)} examples with label issues.")

    predicted_labels = pred_probs.argmax(axis=1)
    label_issues_df["given_label"] = compress_int_array(labels, self.num_classes)
    label_issues_df["predicted_label"] = compress_int_array(predicted_labels, self.num_classes)

    if not save_space:
        if self.label_issues_df is not None and self.verbose:
            print(
                "Overwriting previously identified label issues stored at self.label_issues_df. "
                "self.get_label_issues() will now return the newly identified label issues. "
            )
        self.label_issues_df = label_issues_df
        self.label_issues_mask = label_issues_df[
            "is_label_issue"
        ]  # pointer to here to avoid duplication
    elif self.verbose:
        print(  # pragma: no cover
            "Not storing label_issues as attributes since save_space was specified."
        )

    return label_issues_df

def get_label_issues(self):
    """
    Accessor. Returns `label_issues_df` attribute if previously already computed.
    This ``pd.DataFrame`` describes the label issues identified for each example
    (each row corresponds to an example).
    For column definitions, see the documentation of
    :py:meth:`CleanLearning.find_label_issues<cleanlab.classification.CleanLearning.find_label_issues>`.

    Returns
    -------
    pd.DataFrame
    """

    if self.label_issues_df is None:
        warnings.warn(
            "Label issues have not yet been computed. Run `self.find_label_issues()` or `self.fit()` first."
        )
    return self.label_issues_df
