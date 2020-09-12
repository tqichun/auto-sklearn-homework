#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import os
from pathlib import Path

import openml
import pylab as plt

prefix = "experiment_results2"
summary = {}
for file in os.listdir(prefix):
    dataset_id = int(file.split(".")[0])
    data = json.loads((Path(prefix) / file).read_text())
    name = openml.datasets.get_dataset(dataset_id).name
    summary[name] = data

name2yidx = dict(zip(summary.keys(), range(1, len(summary.keys()) + 1)))
m2style = {
    "origin": {
        "c": "b",
        "marker": "o"
    },
    "use_matrix": {
        "c": "r",
        "marker": "s"
    }
}
plt.rcParams['figure.figsize'] = (15, 9)
for i, metric in enumerate(summary['diabetes']['origin'].keys()):
    plt.subplot(2, 2, i + 1)
    for method_alias, method in [("origin", "l1-knn"), ("use_matrix", "dissim-matrix")]:
        x = []  # metric(performance)
        y = []  # dataset_names
        names = list(summary.keys())
        for name in names:
            x.append(summary[name][method_alias][metric])
            y.append(name2yidx[name])
        plt.yticks(range(1, len(names) + 1), names + [""])
        plt.scatter(x, y, label=method, alpha=0.5, **m2style[method_alias])
        plt.title(metric)
        plt.legend(loc="best")
    plt.grid(alpha=0.5)
plt.suptitle("Comparison of l1-knn and dissim-matrix methods")
plt.rcParams['savefig.dpi'] = 300  # 清晰度
plt.savefig("comparison.png")
print("OK")
