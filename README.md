# Enhancing Medical Image Segmentation with MA-UNet: A Hybrid Multi-scale Attention Framework Based on U-Net

If you want to get the code with the dataset, go to:https://zenodo.org/records/13892731    
![image](https://github.com/user-attachments/assets/461c598b-4ab4-4015-b190-ceaf72de539e)



# Datasets:

- Pulmonary CT lesions (PCL)  is publicly available at https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data/data. Provided by K  Scott Mader, this dataset consists of 267 two-dimensional CT images of lung lesions  and corresponding ground truth images. 
- Digital Database of Thyroid Images (DDTL): This dataset can be publicly  accessed at https://www.kaggle.com/datasets/eiraoi/thyroidultrasound. The DDTL  contains 637 ultrasound images of thyroid nodules and corresponding ground truth  images. 
- Digital Retinal Images for Vessel Extraction (DRIVE): This dataset can be  accessed at https://drive.grand-challenge.org/. It is specifically dedicated to retinal  vessel segmentation. It holds significant importance in the field of medical image  processing, particularly for the study of retinal pathologies such as screening and  diagnosis of diabetic retinopathy (DR).

# Introduction

1. If you want to use our code, you must have the following preparation under the PyTorch framework: see requirement.txt for details

2. Code Guidance: Download the dataset in the above link, put the training images and labels into "data/images" and "data/masks" respectively, and then run the ktrain.py to successfully train our model.
