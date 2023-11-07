# whisper-demo
speech recognition by whisper model, offering a simple training and predict demo code.


# Training
```
python train.py
```

训练损失和学习率变化，可以看到损失降低还是非常明显的，代码中使用了CosineAnnealingLR来降低学习率，
总共训练5个epoch，效果还不错，如果加入更多数据，训练更多轮效果应该会更好。

```tex
Epoch: 1/5: 100%|██████████| 1486/1486 [13:41<00:00,  1.81it/s, lr=1.81e-5, train average loss=0.221, train loss=0.129] 
Epoch: 2/5: 100%|██████████| 1486/1486 [13:41<00:00,  1.81it/s, lr=1.31e-5, train average loss=0.103, train loss=0.379]  
Epoch: 3/5: 100%|██████████| 1486/1486 [13:41<00:00,  1.81it/s, lr=6.91e-6, train average loss=0.045, train loss=0.0376]   
Epoch: 4/5: 100%|██████████| 1486/1486 [13:43<00:00,  1.81it/s, lr=1.91e-6, train average loss=0.0126, train loss=0.000637]
Epoch: 5/5: 100%|██████████| 1486/1486 [13:42<00:00,  1.81it/s, lr=0, train average loss=0.00256, train loss=0.000263] 
```


# Predict
代码中默认使用了单个文件进行predict，可以看到使用fintuned过的模型识别效果还是非常好的。
```
python predict.py
```
> label:
> predict:


