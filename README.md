# Efficient and Effective transfer learning method for Hyperparameter optimizattion.


## Requirements Installation
1. Smac3 needs [SWIG](https://github.com/swig/swig/wiki/Getting-Started) with version >= 3.05. We recommend you to install it manually by:
```
sudo apt-get install build-essential libpcre3-dev
wget http://prdownloads.sourceforge.net/swig/swig-3.0.5.tar.gz
tar xvzf swig-3.0.5.tar.gz
cd swig-3.0.5
./configure && make && sudo make install
```

2. Install all python packages:
```
pip3 install -r requirement.txt --user
```

## Experiment Design
1. on the dimension of source problem.
2. on the dimension of #trial in the source problem.
3. main exp on 30 problems with 100 trials.
4. online scenario: start from zero: avg rank after finishing one problem with a variant #trials.
5. warmstart evaluation.
6. test different #trials in each problem.

## Baselines
1. rs: Random search
2. notl: Independent surrogate: gp and rf.
3. tst: TST-{R/M}
4. sgpr: VIZIER
5. rgpe: RGPE
6. scot: SCOT
7. mklgp: MKL-GP

## TODO
1. test each baseline: get the meta-features of datasets included.
2. implement the online scenario.
3. implement the learning based solutions.
