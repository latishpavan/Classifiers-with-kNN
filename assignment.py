import numpy as np
import matplotlib.pyplot as plt
import math as m
import operator as op
import pandas as pd
import time
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=np.nan) 

def extract_known(filename):
	df=pd.read_csv(filename,index_col=0)
	known=df.loc[df['Game']!='MISSING_VALUE']
	known['Aracade_Game']=known['Aracade_Game'].map({'YES':1,'NO':0})
	known['Adventure_Game']=known['Adventure_Game'].map({'YES':1,'NO':0})
	known['Game']=known['Game'].map({'GAME_1':1,'GAME_3':3,'GAME_2':2,'GAME_4':4})
	known_mat=known.as_matrix()
	return known_mat

def extract_unknown(filename):
	df=pd.read_csv(filename,index_col=0)
	unknown=df.loc[df['Game']=='MISSING_VALUE']
	unknown['Aracade_Game']=unknown['Aracade_Game'].map({'YES':1,'NO':0})
	unknown['Adventure_Game']=unknown['Adventure_Game'].map({'YES':1,'NO':0})
	unknown_mat=unknown.as_matrix()
	return unknown_mat

def classify_group(unknown_mat,known_mat,k):
	for i in range(unknown_mat.shape[0]):
		sqmat=((known_mat[:,:2])-np.tile(unknown_mat[i][:2],(known_mat.shape[0],1)))**2
		sum_mat=np.sum(sqmat,axis=1)
		sort_indices=np.argsort(sum_mat);count={}
		for j in range(k):
			count[known_mat[sort_indices[j]][2]]=count.get(known_mat[sort_indices[j]][2],0)+1
		count_sorted=sorted(count.items(),key=op.itemgetter(1),reverse=True)
		unknown_mat[i][2]=int(count_sorted[0][0])
	return unknown_mat

def extract_level(group_mat):
	df=pd.DataFrame(data=group_mat[:,:],columns=['Aracade_Game','Adventure_Game','Game','Proficiency'])
	known_level=df[df['Proficiency']!='MISSING_VALUE']
	unknown_level=df[df['Proficiency']=='MISSING_VALUE']
	known_level['Proficiency']=known_level['Proficiency'].map({'LEVEL_4':4,'LEVEL_1':1,'LEVEL_2':2,'LEVEL_3':3})
	return known_level.as_matrix(),unknown_level.as_matrix()

def classify_level(known_level,unknown_level,k):
	for i in range(unknown_level.shape[0]):
		sqmat=(known_level[:,:3]-np.tile(unknown_level[i][:3],(known_level.shape[0],1)))**2
		summat=np.sum(sqmat,axis=1)
		sort_indices=np.argsort(summat);count={}
		for j in range(k):
			count[known_level[sort_indices[j]][3]]=count.get(known_level[sort_indices[j]][3],0)+1
		count_sorted=sorted(count.items(),key=op.itemgetter(1),reverse=True)
		unknown_level[i][3]=count_sorted[0][0]
	return unknown_level

def write_to_csv(filled_mat):
	df=pd.DataFrame(data=filled_mat[:,:],columns=['Aracade_Game','Adventure_Game','Game','Proficiency'])
	df['Aracade_Game']=df['Aracade_Game'].map({1:'YES',0:'NO'})
	df['Adventure_Game']=df['Adventure_Game'].map({1:'YES',0:'NO'})
	df['Game']=df['Game'].map({1:'GAME_1',2:'GAME_2',3:'GAME_3',4:'GAME_4'})
	df['Proficiency']=df['Proficiency'].map({1:'LEVEL_1',2:'LEVEL_2',3:'LEVEL_3',4:'LEVEL_4'})
	df.to_csv("final.csv",columns=['Aracade_Game','Adventure_Game','Game','Proficiency'],encoding='utf-8')

start=time.time()
known_inp=extract_known("Assignment_Missing_Value.csv")
unknown_inp=extract_unknown("Assignment_Missing_Value.csv")
found_mat=classify_group(unknown_inp,known_inp,5)
group_mat=np.append(known_inp,found_mat,axis=0)
known_level,unknown_level=extract_level(group_mat)
#print(known_level[:20,:3],unknown_level[:20,:3])
unknown_level=classify_level(known_level,unknown_level,5)
filled_mat=np.append(unknown_level,known_level,axis=0)
write_to_csv(filled_mat)
print(time.time()-start)