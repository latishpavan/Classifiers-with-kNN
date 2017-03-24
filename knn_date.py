import numpy as np
import operator as op
import matplotlib.pyplot as plt

def create_dataset():
	fr=open("dating.txt")
	size=len(fr.readlines())
	mat=np.zeros((size,3))
	labels=[];index=0
	fr=open("dating.txt",encoding='utf-8-sig')
	for line in fr.readlines():
		line_data=line.strip().split('\t')
		line_input=list(map(float,line_data[:3]))
		mat[index]=line_input
		labels.append(line_data[-1])
		index+=1
	return mat,labels

def visualise(mat,labels):
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	col=[];size=[]
	for i in labels:
		if i=='largeDoses':
			col.append(-1.0);size.append(30.0)
		elif i=='smallDoses':
			col.append(0.0);size.append(40.0)
		else:
			col.append(1.0);size.append(50.0)
	ax.scatter(mat[:,0],mat[:,1],c=col,s=size)
	plt.show()

def norm_data(mat):
	min_val=mat.min(0)
	max_val=mat.max(0)
	range_val=max_val-min_val
	norm_mat=mat-np.tile(min_val,(mat.shape[0],1))
	norm_mat=norm_mat/np.tile(range_val,(mat.shape[0],1))
	return norm_mat,range_val,min_val

def classify(mat,inX,range_val,min_val,k,labels):
	inX=(inX-min_val)/range_val
	diffmat=mat-np.tile(inX,(mat.shape[0],1))
	sqmat=diffmat**2
	summat=np.sum(diffmat,axis=1)
	finmat=summat**0.5
	sorted_list=finmat.argsort()
	pref_dict={}
	for i in range(k):
		pref_dict[labels[sorted_list[i]]]=pref_dict.get(labels[sorted_list[i]],0)+1
	pref_dict=sorted(pref_dict.items(),key=op.itemgetter(1),reverse=True)
	return pref_dict[0][0]

mat,labels=create_dataset()
#visualise(mat,labels)
norm_mat,range_val,min_val=norm_data(mat);error=0;tot=0
fr=open("test_dating.txt",encoding='utf-8-sig')
for line in fr.readlines():
	line_input=line.split('\t')
	original_answer=line_input[-1]
	inX=list(map(float,line_input[:3]))
	k=classify(norm_mat,inX,range_val,min_val,3,labels)
	#print("original answer is {0} \t Predicted answer is {1}".format(original_answer,k))
	if k!=original_answer:
		error+=1
	print(k,original_answer)	
	tot+=1
print("Error is ",tot)