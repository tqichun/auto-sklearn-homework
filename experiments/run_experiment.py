# -*- encoding: utf-8 -*-
"""
==============
Classification
==============

The following example shows how to fit a simple classification model with
*auto-sklearn*.
"""
import json
import os
from pathlib import Path

import click
import openml
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from joblib import Parallel, delayed
import autosklearn.classification


############################################################################
# Data Loading
# ============
def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    tasks_a = openml.tasks.list_tasks(task_type_id=1, status='active')
    tasks_a = pd.DataFrame.from_dict(tasks_a, orient="index")[['tid', 'did', 'name', 'target_feature']]
    target = tasks_a.query(f'did == {dataset_id}')['target_feature'].tolist()[0]
    df, y, cat, _ = dataset.get_data(target=target)
    cat_cols = df.select_dtypes(['category']).columns
    df[cat_cols] = df[cat_cols].astype(object)
    obj_cols = df.select_dtypes(['object']).columns
    for col in obj_cols:
        mask = ~pd.isna(df[col])
        df[col][mask] = LabelEncoder().fit_transform(df[col][mask])
    y = LabelEncoder().fit_transform(y)
    df = df.astype(float)
    return df.values, y, cat

def run_single_experiment(dataset_id):
    info = {}
    X, y, cat = load_dataset(dataset_id)
    is_binary = type_of_target(y) == "binary"
    feat_type = ["Categorical" if c else "Numerical" for c in cat]

    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    for mode in ["origin", "use_matrix"]:
        print(f"current experiment mode = {mode}")
        ############################################################################
        # Build and fit a classifier
        # =========================
        os.system('rm -rf /tmp/autosklearn_classification_example*')
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
            tmp_folder='/tmp/autosklearn_classification_example_tmp',
            output_folder='/tmp/autosklearn_classification_example_out',
        )
        dataset_name = str(dataset_id) if mode == "use_matrix" else "unk"
        print(f"dataset_name = {dataset_name}")
        automl.fit(X_train, y_train, dataset_name=dataset_name, feat_type=feat_type)
        ############################################################################
        # Print the final ensemble constructed by auto-sklearn
        # ====================================================

        print(automl.show_models())

        ###########################################################################
        # Get the Score of the final ensemble
        # ===================================

        predictions = automl.predict(X_test)
        info[mode] = {}
        info[mode]["accuracy_score"] = sklearn.metrics.accuracy_score(y_test, predictions)
        if is_binary:
            info[mode]["recall_score"] = sklearn.metrics.recall_score(y_test, predictions)
            info[mode]["f1_score"] = sklearn.metrics.f1_score(y_test, predictions)
            info[mode]["precision_score"] = sklearn.metrics.precision_score(y_test, predictions)
    Path(f"experiment_results/{dataset_id}.json").write_text(json.dumps(info, indent=4))

@click.command()
@click.option("--dataset-id", "-d")
def main(dataset_id):
    Path("experiment_results").mkdir(parents=True, exist_ok=True)
    run_single_experiment(int(dataset_id))

if __name__ == '__main__':
    main()


