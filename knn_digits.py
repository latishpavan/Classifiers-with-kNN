import numpy as np
import operator as op
import matplotlib.pyplot as plt
import cv2
np.set_printoptions(threshold=np.nan)

def file2vector(filename):
	fr=open(filename)
	bin_mat=np.zeros(1024);i=0
	for line in fr.readlines():
		for j in range(32):
			bin_mat[i+j]=int(line[j])
		i+=32
	return bin_mat

'''img=cv2.imread("1.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img=cv2.resize(gray_img,(32,32))
np.set_printoptions(threshold=np.nan)
print(gray_img<100)'''

def classify(bin_mat=None,k=None,labels=None):
	curr_dig=0;curr_dig_file=0;
	curr_file_name='train/'+str(curr_dig)+'_'+str(curr_dig_file)+'.txt'
	print(file2vector(curr_file_name))

classify()
