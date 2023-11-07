# Whisper-demo
speech recognition by whisper model, offering a simple training and predict demo code.


# Dataset Download
> 数据集下载地址`https://www.kaggle.com/datasets/bryanpark/chinese-single-speaker-speech-dataset`

数据集的目录结构：
```
chinese-single-speaker-speech-dataset/call_to_arms/
chinese-single-speaker-speech-dataset/chao_hua_si_she/
chinese-single-speaker-speech-dataset/transcript.txt
```

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
```
label:   《呐喊》自序·鲁迅·我在年青时候也曾经做过许多梦
predict: 《呐喊》自序·鲁迅·我在年青时候也曾经做过许多梦
```


# Wer
由于demo数据集中验证数据没有给label数据，只提供了wav的数据，所以无法验证wer，但是通过手动单个测试发现5个epoch的训练效果也是非常好的,
这里提供一个计算wer的demo code用于用户去计算wer；
```
import evaluate

metric = evaluate.load("wer")
wer = metric.compute(predictions="我是中国仁", references="我是中国人")
print(wer)  # 0.2
```

