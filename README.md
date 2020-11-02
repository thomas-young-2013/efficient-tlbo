# TOPO: Efficient and Scalable Model Re-optimizations via Two-Phase Transfer Learning


## Requirements Installation
1. [SWIG](https://github.com/swig/swig/wiki/Getting-Started) >= 3.05. If not, we recommend you to install it manually by:
```bash
apt-get install swig3.0
ln -s /usr/bin/swig3.0 /usr/bin/swig
```

2. Install all python packages by:
```bash
pip install -r requirement.txt --user
```
If failed, try installing Numpy, Scipy and Cython first.

## Benchmark
The experiment data used in this paper is available at Google drive now. ([download link here](https://drive.google.com/drive/folders/1gRKT3uIOiLs8I2TiZSlBj2LWbNlqI5ya))
The zipped file contains 451 files (450 files including history observations of 4 ML algorithms on 30 datasets + 1 file storing the meta-features of 30 datasets).
To use the benchmark data, please make directory *./data/hpo_data* in the root directory of this project and move all the pickle files into the new folder.
***

30 OpenML datasets used in the benchmark are as follows:
Name | Rows | Columns | Categories
--- | - | - | - 
balloon | 2001 | 1 | 2 
kc1 | 2109 | 21 | 2 
quake | 2178 | 3 | 2 
segment | 2310 | 19 | 7 
madelon | 2600 | 500 | 2 
space_ga | 3107 | 6 | 2 
splice | 3190 | 60 | 3 
kr-vs-kp | 3196 | 36 | 2 
sick | 3772 | 29 | 2 
hypothyroid(1) | 3772 | 29 | 4 
hypothyroid(2) | 3772 | 29 | 2 
pollen | 3848 | 5 | 2 
analcatdata_supreme | 4052 | 7 | 2 
abalone | 4177 | 8 | 26 
spambase | 4600 | 57 | 2 
winequality_white | 4898 | 11 | 7 
waveform-5000(1) | 5000 | 40 | 3 
waveform-5000(2) | 5000 | 40 | 2 
page-blocks(1) | 5473 | 10 | 5 
page-blocks(2) | 5473 | 10 | 2 
optdigits | 5610 | 64 | 10 
satimage | 6430 | 36 | 6 
wind | 6574 | 14 | 2 
musk | 6598 | 167 | 2 
delta_ailerons | 7129 | 5 | 2 
mushroom | 8124 | 22 | 2 
puma8NH | 8192 | 8 | 2 
cpu_small | 8192 | 12 | 2 
cpu_act | 8192 | 21 | 2 
bank32nh | 8192 | 32 | 2 

***

The benchmark includes:

1. The validation balanced accuracy of <20k configurations (10k sampled by random search + 
<10k sampled by Bayesian Optimization) for each algorithm on each dataset. 
Saved as a dictionary **<configuration, balanced accuracy>** in *{dataset_name}-{algorithm_name}-random-20000.pkl*
2. The meta-features of 30 datasets.
 Saved as a dictionary **<dataset_name, meta-feature>** in *dataset_metafeatures.pkl*

## Baselines
1. rs: Random search
2. notl: Independent GP and Random Forest.
3. tst: TST-{R/M}
4. sgpr: VIZIER
5. rgpe: RGPE
6. scot: SCOT
7. gogpe: GOGPE

## Experiment Design and Reproduction Scripts

1. Offline scenario (E.g. Random Forest on RGPE and TOPO using 5 source tasks)

   ```bash
   python tools/offline_benchmark.py --trial_num 75 --num_random_data 20000 --methods rgpe,topo --algo_id random_forest --num_source_problem 5
   ```

2. Online scenario (E.g. Random Forest on RGPE and TOPO)

   ```bash
   python tools/online_benchmark.py --trial_num 50 --num_random_data 20000 --methods rgpe,topo --algo_id random_forest
   ```

3. Source knowledge learning (E.g. Random Forest)

   ```bash
   python tools/studies/source_extraction.py --algo_id random_forest
   ```

4. Target weight analysis (E.g. Random Forest)

   ```bash
   python tools/offline_benchmark.py --trial_num 75 --num_random_data 20000 --methods rgpe,topo --algo_id random_forest --save_weight true
   ```

5. Surrogate combination strategy (E.g. Random Forest)

   ```bash
   python tools/studies/gp_fusion.py --algo_id random_forest
   ```

6. Runtime analysis

   The same as 1 or 4

