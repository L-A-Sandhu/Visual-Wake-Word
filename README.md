# Visual-Wake-Word
Visual wake word is a simple case where we have to detect the presence of a person inside an image frame. The rest of the repository is divided as follows.
  1. Data-Set Prepration 
  2. Mobile Net 
  3. Inception Net
  
## Data Set Prepration 
 The dataset used in this case is actually drived from the MS coco 2014 dataset. Although it is a huge dataset  however we have used Yolov3 to divide images in to two classes named as 
 1. Person 
 2. No-Person

To divide the actual data set in these two classes with the help of yolov3 we have made some changes in **detect.py**. Follow these steps to prepare the dataset.
```
git clone 
cd Visual-Wake-Word/
cd data-prep/
conda create env -n <environment -name> python==3.9.12
conda activate <environment-name>
pip install -r requirements.txt
python detect.py --source <data-set path> --resize=96

```
Now you can see new folder name data inside the main repository.
## Mobile Net
In this section we will explain traning, testing and infrence steps for Mobile Net. please follow the following commands 
```
cd ../
conda create env -n <environment -name> python==3.7.4
conda activate <environment-name>
pip install -r requirements.txt
cd M0bile_Net/
```
### Traning 
```
python Mobile-Net.py  --model_dir=<Location for saving model> --data=<data location> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=train --b_s=16 --e=100
```
### Test 
```
python Mobile-Net.py  --model_dir=<Location for saving model> --data=<data location> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=test --b_s=16 --e=100

```
### Inference 
```
python Mobile-Net.py  --model_dir=<Location for saving model> --data=<data location> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=infer --b_s=16 --e=100
```
## Inception-Net
In this section we will explain traning, testing and infrence steps for Inception Net. please follow the following commands 
```
cd ../Inception_NET/
```
### Traning 
```
python Inception-Net.py  --model_dir=<Location for saving model> --data=<data location> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Inception-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=train --b_s=16 --e=100
```
### Test 
```
python Inception-Net.py  --model_dir=<Location for saving model> --data=<data location> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Inception-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=test --b_s=16 --e=100

```
### Inference 
```
python Inception-Net.py  --model_dir=<Location for saving model> --data=<data location> --inp=<train , test or infer> --b_s=< Batch size> --e=<epoch>
example command 
python Inception-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=infer --b_s=16 --e=100
```




