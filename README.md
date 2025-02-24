# SRNE

This code is for Evolving Form and Function: Dual-Objective Optimization in Neural Symbolic Regression Networks, published in GECCO '24. Since publication, an error was discovered in the code underpinning this manuscript. The error has been rectified here. But, the results have not, to date, replicated.

The error was with how the tokenized target was fed into the model during training. By standard, the first start-of-sequence token is removed and then the tokens are masked so the current predicted token only has access to previously predicted tokens. In the previous code, the first start-of-sequence token was not properly removed, leading to data leakage and falsely good results from pretrained models.

At present, the pretrained models are overfitting during training, leading to models that are unable to predict valid equations. For dual-objective optimization, the initial pretraining is necessary to produce equations that are valid (can be evaluated on X values to produce Y-hat values) in order to improve network performance both symbolically and numerically.

To run, first 25 models pretrained with backpropagation need to be generated.  This allows for a pool of individuals for the initial parent population to be chosen from randomly. To do so, run the following 25 times with a different unique ID (last number) from 0-24 each time. When done, there should be 25 saved models in the pretrain/Pretrained folder, encoder0_best - encoder24_best:

```
python pretrain/run_srkit.py numEpochs uniqueID
```

Then, run SRNE evolution, again specifying a unique ID each run:
```
python run_srkit.py uniqueID
```
