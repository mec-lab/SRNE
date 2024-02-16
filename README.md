# SRNE

To run, first 25 models pretrained with backpropagation need to be generated. To do so, run the following 25 times with a different unique ID (last number) from 0-24 each time. When done, there should be 25 saved models in the pretrain/Pretrained folder, encoder0_best - encoder24_best:

```
python pretrain/run_srkit.py 50 0
```

Then, run SRNE evolution, again specifying a unique ID each run:
```
python run_srkit.py 0
```
