The source code of paper Rethinking Missing Data: Aleatoric Uncertainty-Aware Recommendation

The training steps are divided into two stages.

The first stage is backbone model training step:

python main.py --data=globo --neg_sample_u=0.1 --model=MF --cuda --stage=backbone 

The second stage is uncertain model training step:

python main.py --data=globo --neg_sample_u=1.0 --model=MF --cuda --stage=uncertain --beta=0.01 --gamma=0.01
