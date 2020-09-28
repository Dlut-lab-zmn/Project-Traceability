Running Mode:

We only choose 64 (args.batch_size) blocks (48*48*3, the top-left region) from the original image to attack. 

1. Origin model(default)
python attack.py --model ./model/origin.tar
2. Model with noise
python attack.py --model ./model/noise1.tar

Attack method:

1. PGD(Default)
python attack.py --model ./model/noise1.tar 

2. L2PGD
python attack.py --model ./model/noise1.tar --L2PGD


Concat image:
python attack.py --model ./model/noise1.tar --L2PGD  --concat

If you don't set the --concat, the program will store all blocks.

ID(Default 0):
used for store the related results

python attack.py --model ./model/noise1.tar --L2PGD  --concat --id 0

The test models and datas are available at 

link：https://pan.baidu.com/s/1VIJ7TFPxl_tUYmMTDAjJMA 
password：dli6 
