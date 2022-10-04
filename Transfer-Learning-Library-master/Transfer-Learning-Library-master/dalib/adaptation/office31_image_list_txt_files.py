import numpy as np
import pandas as pd
import os
import cv2

img_list_dir = 'C:\\Users\\Nick\\Desktop\\WPI\\ARL\\Domain_Adaptation\\Original_images\\Original_images\\Original_images_names.csv'

img_list = pd.read_csv(img_list_dir)

amz_img = img_list['Amz_Img'].dropna()
amz_lbl = img_list['Amz_Label'].dropna()
dslr_img = img_list['DSLR_Img'].dropna()
dslr_lbl = img_list['DSLR_Label'].dropna()
wbcm_img = img_list['Webcam_Img'].dropna()
wbcm_lbl = img_list['Webcam_Label'].dropna()

# print(img_list)
# print()
# print()
# print(img_list['Webcam_Label'].dropna())
# print(len(amz_img))
# print(len(amz_lbl))
# print(len(dslr_img))
# print(len(dslr_lbl))
# print(len(wbcm_img))
# print(len(wbcm_lbl))