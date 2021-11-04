Building
--------

```bash
. ./build_docker.sh
```

Running
-------

```bash
. ./run_train.sh
```

Tensorboard
-----------

```bash
. ./run_tensorboard.sh
```

#### working on the repo
```bash
 # always rebuild (the docker image to reflect code changes):
 . ./build_docker.sh && . ./run_train.sh && docker ps
 
 # always restart (tensorboard, as it will run in docker): 
 . ./run_tensorboard.sh 
```
Try out training with a real Dataset
------------------------------------
```bash
# Run the train file with Dataset
python3 Train_complex_DataSet_with_CNN.py

# Run Tensorboard 
tensorboard --bind_all --logdir logs
```
Run pytest locally
------------------------------------
```bash
# to run all tests together
pytest -v 

# to run a specific test using custom marker "dataset"
pytest -v -m dataset  
```
Run pytests in the docker container
------------------------------------
```bash
# run all test together with showing the 
# coverage precentage of the tested code
./run_pytest.sh

# run dataSet test only
./run_dataSet_test.sh

# run preprocessing test only
./run_preprocessing_test.sh



```