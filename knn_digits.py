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

def img2mat(filename):
	img=cv2.imread(filename)
	gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray_img=cv2.resize(gray_img,(32,32))
	
def classify(bin_mat,k=None):
	curr_dig=0;curr_dig_file=0;end_num=180;dist_mat=np.array([])
	while curr_dig<10:
		curr_file_name='train/'+str(curr_dig)+'_'+str(curr_dig_file)+'.txt'
		test_mat=file2vector(curr_file_name)
		sqmat=(bin_mat-test_mat)**2
		fin_mat=np.sum(sqmat)
		dist_mat=np.append(dist_mat,fin_mat)
		curr_dig_file+=1
		if curr_dig_file==end_num:
			curr_dig+=1;curr_dig_file=0
	indices=np.argsort(dist_mat);dist={}
	for i in range(k):
		dist[int(indices[i]/180)]=dist.get(int(indices[i]/180),0)+1
	sorted_dict=sorted(dist.items(),key=op.itemgetter(1),reverse=True)
	print("The predicted digit is ",sorted_dict[0][0])

classify(file2vector("test/txt.2_22"),20)
