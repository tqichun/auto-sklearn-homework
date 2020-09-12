#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
from pathlib import Path

import openml
import pandas as pd

tasks_a = openml.tasks.list_tasks(task_type_id=1, status='active')
tasks_a = pd.DataFrame.from_dict(tasks_a, orient="index")
df = tasks_a.query("NumberOfClasses == 2 and NumberOfFeatures < 10 and NumberOfFeatures >=2 and NumberOfInstances <1000 and NumberOfInstances >500")
names = ['diabetes', 'monks-problems-1', 'strikes', 'kdd_el_nino-small', 'disclosure_z', 'balance-scale']
dataset_ids=[]
for dataset_id,row in df.iterrows():
    name = row['name']
    if name in names:
        print(name)
        names.remove(name)
        dataset_ids.append(row["did"])

Path("wanted_datasets.json").write_text(json.dumps(dataset_ids))
