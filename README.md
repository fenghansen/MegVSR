# MegVSR
 For the Video Super-Resolution competition of Megvii
## Environment  
 Megengine 0.5.1 / Pytorch 1.3.1
## Method  
| Name | Academy | score |  
| - | - | - |  
9.03 | SRResnet | 30.086dB  
9.04 | SR-RRDB(6) | 30.296dB  
9.08 | 完成torch到megengine的转换 | 30.295dB  
9.10 | 完成类Unet的Encoder转换，VRRDB(3) | 30.486dB  
9.11 | 拓展为5帧的版本VRRDB(5) | 30.576dB  
9.18 | RRDB版SlowFusion带残差 | 30.407dB  
9.19 | RRDB版SlowFusion不带残差 | 30.213dB  
9.24 | 完成基于光流法5帧EarlyFusion的模型，训练一天 | 30.583dB  

来不及了，随便试试。GG。  
https://studio.brainpp.com/competition?tab=rank  
