import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.utils
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from ....util.logging_ import get_logger


class KNearestDatasets(object):
    def __init__(self, metric='l1', random_state=None, metric_params=None):
        self.logger = get_logger(__name__)

        self.metric = metric
        self.model = None
        self.metric_params = metric_params
        self.metafeatures = None
        self.runs = None
        self.best_configuration_per_dataset = None
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.scaler = MinMaxScaler()

        if self.metric_params is None:
            self.metric_params = {}

    def fit(self, metafeatures, runs):
        """Fit the Nearest Neighbor model.

        Parameters
        ----------
        metafeatures : pandas.DataFrame
            A pandas dataframe. Each row represents a dataset, each column a
            metafeature.
        runs : dict
            Dictionary containing a list of runs for each dataset.
        """
        assert isinstance(metafeatures, pd.DataFrame)
        assert metafeatures.values.dtype in (np.float32, np.float64)
        assert np.isfinite(metafeatures.values).all()
        assert isinstance(runs, pd.DataFrame)
        assert runs.shape[1] == metafeatures.shape[0], \
            (runs.shape[1], metafeatures.shape[0])

        self.metafeatures = metafeatures
        self.runs = runs
        self.num_datasets = runs.shape[1]

        # Fit the metafeatures for scaler
        self.scaler.fit(self.metafeatures)

        # for each dataset, sort the runs according to their result
        best_configuration_per_dataset = {}
        for dataset_name in runs:
            if not np.isfinite(runs[dataset_name]).any():
                best_configuration_per_dataset[dataset_name] = None
            else:
                configuration_idx = runs[dataset_name].index[
                    np.nanargmin(runs[dataset_name].values)]
                best_configuration_per_dataset[dataset_name] = configuration_idx

        self.best_configuration_per_dataset = best_configuration_per_dataset

        if callable(self.metric):
            self._metric = self.metric
            self._p = 0
        elif self.metric.lower() == "l1":
            self._metric = "minkowski"
            self._p = 1
        elif self.metric.lower() == "l2":
            self._metric = "minkowski"
            self._p = 2
        else:
            raise ValueError(self.metric)

        self._nearest_neighbors = NearestNeighbors(
            n_neighbors=self.num_datasets, radius=None, algorithm="brute",
            leaf_size=30, metric=self._metric, p=self._p,
            metric_params=self.metric_params)

    def kNearestDatasets(self, x, k=1, return_distance=False, dataset_name=None):  # add dataset_name
        """Return the k most similar datasets with respect to self.metric

        Parameters
        ----------
        x : pandas.Series
            A pandas Series object with the metafeatures for one dataset

        k : int
            Number of k nearest datasets which are returned. If k == -1,
            return all dataset sorted by similarity.

        return_distance : bool, optional. Defaults to False
            If true, distances to the new dataset will be returned.

        Returns
        -------
        list
            Names of the most similar datasets, sorted by similarity

        list
            Sorted distances. Only returned if return_distances is set to True.
        """
        matric_csv_path = Path(__file__).parent / "csvDataDissimMatrix(dataset).csv"
        task2dataset_path = Path(__file__).parent / "task2dataset.json"
        task2dataset = json.loads(Path(task2dataset_path).read_text())
        task_ids = pd.Series(self.metafeatures.index)
        dataset_ids = task_ids.map(task2dataset)
        dataset_ids.dropna(inplace=True)
        matrix = pd.read_csv(matric_csv_path, index_col=0)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.index
        valid_dataset_ids = np.intersect1d(dataset_ids, matrix.index) # 77个交集
        matrix = matrix.loc[valid_dataset_ids, :]
        valid_dataset_name = np.unique(matrix.columns) # 800个
        if not isinstance(dataset_name, str):
            assert ValueError(f'dataset_name {dataset_name} should be str type.')
        if dataset_name not in valid_dataset_name:
            # print(f"[WARNING] dataset_name {dataset_name} do not exist in csvDataDissimMatrix(dataset).csv, "
            #       f"using default method")
            # dataset_name = valid_dataset_name.tolist()[np.random.randint(0, valid_dataset_name.size)]
            X_train = self.scaler.transform(self.metafeatures)
            x = x.values.reshape((1, -1))
            x = self.scaler.transform(x)
            self._nearest_neighbors.fit(X_train)
            if k==-1:
                k=X_train.shape[0]
            distances_, neighbor_indices = self._nearest_neighbors.kneighbors(
                x, n_neighbors=k, return_distance=True)

            assert k == neighbor_indices.shape[1]

            rval = [self.metafeatures.index[i]
                    # Neighbor indices is 2d, each row is the indices for one
                    # dataset in x.
                    for i in neighbor_indices[0]]

            if return_distance is False:
                return rval
            else:
                return rval, distances_[0]
        assert type(x) == pd.Series
        # self._nearest_neighbors.fit(X_train)
        arr = np.array(matrix[dataset_name])
        if arr.ndim >= 2:
            arr = arr[:, 0]
        rval = np.argsort(arr)
        distances = np.array(arr)
        distances.sort()
        rval = self.metafeatures.index[rval]
        if return_distance is False:
            return rval
        else:
            return rval, distances

    def kBestSuggestions(self, x, k=1, exclude_double_configurations=True, dataset_name=None):  # add dataset_name
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        nearest_datasets, distances = self.kNearestDatasets(x, -1,
                                                            return_distance=True,
                                                            dataset_name=dataset_name)  # add dataset_name

        kbest = []

        added_configurations = set()
        for dataset_name, distance in zip(nearest_datasets, distances):
            best_configuration = self.best_configuration_per_dataset[
                dataset_name]

            if best_configuration is None:
                self.logger.warning("Found no best configuration for instance "
                                    "%s" % dataset_name)
                continue

            if exclude_double_configurations:
                if best_configuration not in added_configurations:
                    added_configurations.add(best_configuration)
                    kbest.append((dataset_name, distance, best_configuration))
            else:
                kbest.append((dataset_name, distance, best_configuration))

            if k != -1 and len(kbest) >= k:
                break

        k = 25
        return kbest[:k]
