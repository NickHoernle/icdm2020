Code for reproducing the experiments from the [The Phantom Steering Effect in Q&A Websites](https://arxiv.org/pdf/2002.06160.pdf)

# Installation
Create new virtual environment:
```
conda create -n so python=3.6.9
source activate so
```

Install the package: from the root of this directory run:
```
pip install -e .
```

# Get data
The raw data must be present before any of the code is run. For the votong data, this cannot be obtained and thus these plots cannot be reproduced. However, the analysis on the editing data can be reproduced and this data can be queried and downloaded from the SO Data Query website.

The data can be downloaded at [this link](https://data.stackexchange.com/stackoverflow/query/1216727/so-badges-work-count-edits-per-day-for-strunk-and-white-users) where both the `EXECUTE sp_executesql @sql;` and the `SELECT UserId, Date FROM #successfulUsers;` commands should be run.

The individual files should be joined into one source file for data and can be prepared for use via the following script:
```
```

# Running code
The modeling code is run as follows:
```
run_inference.py --input data/pt_editor --output experiments/StrunkWhite --target-badge StrunkWhite --epochs 2000 --early-stopping-lim 250 --model-name full_personalised_normalizing_flow --quiet --no-cuda --lr 0.0001 --gamma 0.99
```

(The experimental hyperparameter search is in the `scripts/experiment.txt` file)
