# Visual-Wake-Word
Visual wake word is a simple case where we have to detect the presence of a person inside an image frame. Please clone the repo using the following commands 
```
git clone https://github.com/L-A-Sandhu/Visual-Wake-Word.git
```
The rest of the repository is divided as follows.
  1. Requirements
  2. Data-Set Prepration 
  3. Fine Tunning 
  4. Pre-trained Model
  5. Results 

## Requirements 
This repository requires 
* **tensorflow**
* **matplotlib**
* **scipy**
* **protobuf**


This work requires to build two different environments one for data prepration and other for training and testing of Mobile Net and inception net on the preapared data. 
### Building Environment for data prepration 
please follow the following set of commands 
```
cd Visual-Wake-Word/
cd data-prep/
conda create env -n <environment -name> python==3.9.12
conda activate <environment-name>
pip install -r requirements.txt
```
### Building Environment for Testing and Traning 
Please follow the following set of ommands 
```
cd ../
conda create env -n <environment -name> python==3.7.4
conda activate <environment-name>
pip install -r requirements.txt

```
## Data Set Prepration 
 The dataset used in this case is actually drived from the MS coco 2014 dataset. Although it is a huge dataset  however we have used Yolov3 to divide images in to two classes named as 
 1. Person 
 2. No-Person

To divide the actual data set in these two classes with the help of yolov3 we have made some changes in **detect.py**. Follow these steps to prepare the dataset.
```
conda activate < Environment for data prepration >
cd ./data-prep/
python detect.py --source <data-set path> --resize=96

```
Now you can see new folder name **data** inside the main repository.
## Fine Tunning 
The keras implementation of Mobile and Inception Net trained these models imagenet dataset. However, to train these model on coustom data set we may use transfer learning . In the comming section we be explaining the training and testing of Mobile and well and Incepion Ne.

```
cd../<M0bile_Net or Inception_NET>/

```
#### Tuning
```
python <Mobile-Net.py or Inception_NET>  --model_dir=<Location for saving model> --data=<data location> --inp=<tune> --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=tune --b_s=16 --e=100
```
#### Testing  
```
python <Mobile-Net.py or Inception_NET>  --model_dir=<Location for saving model> --data=<data location> --inp=<test> 
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=test
````
#### Training 
```
python <Mobile-Net.py or Inception_NET>  --model_dir=<Location for saving model> --data=<data location> --inp=<train > --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=train --b_s=16 --e=100
```
### Resume
```
python <Mobile-Net.py or Inception_NET>  --model_dir=<Location for saving model> --data=<data location> --inp=<resume> --b_s=< Batch size> --e=<epoch>
example command 
python Mobile-Net.py  --model_dir='./checkpoint/' --data='../data/' --inp=resume --b_s=16 --e=100

```

## Pretrained Model
Orignally the keras fram work has trained Mobile and Inception Net implementations on image-net dataset. This work has fine tunned it on coustom dataset to find the exsisitance of a person with in an image. You can use this trained model by simpley downloading their checkpoints from the links given  below and placing them in side the folders **M0bile_Net** and **Inception_NET** respectively. 
```
https://drive.google.com/file/d/1jJ4ZxF5q5tkrmgjVziXJhN3UD21470w9/view?usp=sharing
https://drive.google.com/file/d/1vi4KFKIRsRQ_dkuU90EWfpKd4JxvjHO7/view?usp=sharing

```


## Results and Comparisons 
The comparision between Mobile and Inception Net is shown in the following table 

| Model         | Parameter | Latency| Accuracy| Size on disk(MB) | FLops |
|---------------|-----------|--------|---------|------------------|-------|
| Inception-Net | 22,114,850|0.00138 | 0.72    | 273.0            |0.681 G|
| Mobile-Net    | 3,783,510 |0.00091 | 0.74    | 49.8             |0.116 G|


Confusion matix for Mobile Net 

| Confusion Matrix  | Person | No-Person |
|-------------------|--------|-----------|
| Person            | 4613   | 918       |
| No-Person         | 1541   | 2958      |



Confusion Matrix for Inception Net 

| Confusion Matrix  | Person | No-Person |
|-------------------|--------|-----------|
| Person            | 3061   | 2740      |
| No-Person         | 292    | 4207      |


