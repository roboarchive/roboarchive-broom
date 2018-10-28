# roboarchive-broom
A toolbox to clean archive documents

## Install dependencies

1. Python 3.6
2. Keras
3. OpenCV


## Run the script

```bash
python3.6 train_cnn.py
```

## Check results

```bash
tensorboard --logdir tensorboard_log --reload_interval 1
```


## copy training data

```bash
gsutil -m cp -nr gs://robo-broom/data/train/ .
```
