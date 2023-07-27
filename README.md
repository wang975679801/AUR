# Rethinking Missing Data: Aleatoric Uncertainty-Aware Recommendation

This repository contains the source code for the paper "Rethinking Missing Data: Aleatoric Uncertainty-Aware Recommendation".

## Training Steps

The training steps are divided into two stages.

### 1. Backbone Model Training Step

To train the backbone model, run the following command:

```bash
python main.py --data=globo --neg_sample_u=0.1 --model=MF --cuda --stage=backbone
```
### 2. Uncertain Model Training Step
To train the uncertain model, run the following command:
```bash
python main.py --data=globo --neg_sample_u=1.0 --model=MF --cuda --stage=uncertain --beta=0.01 --gamma=0.01
```
Adjust the parameters (neg_sample_u, beta, gamma, etc.) as per your requirements.
