# Robust transfer learning (RTL) method for Bayesian optimization


## Requirements Installation
1. Smac3 needs [SWIG](https://github.com/swig/swig/wiki/Getting-Started) with version >= 3.05. We recommend you to install it manually by:
```
sudo apt-get install build-essential libpcre3-dev
wget http://prdownloads.sourceforge.net/swig/swig-3.0.5.tar.gz
tar xvzf swig-3.0.5.tar.gz
cd swig-3.0.5
./configure && make && sudo make install
```

2. Dependent package `George` needs eigen3 support:
```
sudo apt install libeigen3-dev
```

3. Install all python packages:
```
pip3 install -r requirement.txt --user
```

## Running arguments
1. svm metadata: `-f data/svm/ -dataset A9A -tries 10 -s tst-r -bandwidth 0.1 -hpRange 6 -hpIndicatorRange 3`
2. weka metadata: `-f data/weka/ -dataset bands -tries 10 -s tst-r -bandwidth 0.9 -hpRange 103 -hpIndicatorRange 64`

## Experiment Design
1. Divide train/validation/test dataset to get the validation average rank and the test average rank.
2. Computed 1) the average rank score in each iteration for each problem; 2) the average distance to the global minimum.
3. Compare the average rank with the growth of datasets.
3. Grid hyperparameter performance due to the small size of hyperspace.
4. Exhaustive grid search is not possible at all due to the higher number of hyperparameters.
5. On the methodological side, a comprehensive database of learning problems should be compiled, progressively closing the gap between benchmark problems and real-world applications.
6. Theoretic guarantee.

## Baselines
1. Random search
2. Independent surrogate: gp and rf.
3. TST-{R/M}
4. VIZIER
5. RGPE
6. SCOT (collaborative, omitted)
7. MKL
