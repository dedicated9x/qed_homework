Step 1. Environment installation.
```
conda env create -f env.yaml
```

Step 2. Dataset downloading. <br>
It should be done manually. <br>
Expected result is as follows 
```
data/
|-- cybersecurity_test.csv
|-- cybersecurity_training.csv
```

Step 3. Training. <br>
In the case of gradient boosting:
```
python train_gbm.py
```
or in the case of neural network:
```
python train_nn.py
```