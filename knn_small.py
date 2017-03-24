import numpy as np
import operator as op

def create_dataset():
	data=np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return data,labels

def classify(data,groups,inX,k):
	data_set_size=data.shape[0]
	diffmat=np.tile(inX,(data_set_size,1))-data
	sqmat=diffmat**2
	sqdit=np.sum(sqmat,axis=sqmat.ndim-1)
	distances=sqdit**0.5
	ind=distances.argsort()
	classifier={}
	for i in range(k):
		classifier[groups[ind[i]]]=classifier.get(groups[ind[i]],0)+1
	classifier=sorted(classifier.items(),key=op.itemgetter(1),reverse=True)
	return classifier[0][0]

data,groups=create_dataset()
print(classify(data,groups,np.array([2,2]),3))
