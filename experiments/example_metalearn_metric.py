# -*- encoding: utf-8 -*-
"""
==============
Classification
==============

The following example shows how to fit a simple classification model with
*auto-sklearn*.
"""
import os

import openml
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder

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


dataset_id = 2

X, y, cat = load_dataset(dataset_id)

feat_type = ["Categorical" if c else "Numerical" for c in cat]

X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

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
automl.fit(X_train, y_train, dataset_name=str(dataset_id), feat_type=feat_type)

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

print(automl.show_models())

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("acc score:", sklearn.metrics.accuracy_score(y_test, predictions))
print("recall score:", sklearn.metrics.recall_score(y_test, predictions))
print("f1 score:", sklearn.metrics.f1_score(y_test, predictions))
