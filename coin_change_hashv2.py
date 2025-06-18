# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:35:44 2025

@author: e_kle
"""



# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:06:16 2025

@author: hkleikamp
"""

#%% Modules
from inspect import getsourcefile

import warnings

import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np
from collections import Counter
import time

# %% change directory to script directory (should work on windows and mac)
basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())


# %% Get elemental metadata (or replace with utils table)
mass_table = str(Path(basedir, "mass_table.tsv"))           # table containing element masses, default: CartMFP folder/ mass_table.tsv"
emass = 0.000548579909  # electron mass
mdf = pd.read_csv(mass_table, index_col=[0], sep="\t")
mdf.loc["+"]=-emass
mdf.loc["-"]=+emass

#%% Functions



#parse form
def parse_form(form): #chemical formular parser
    e,c,comps="","",[]
    for i in form:
        if i.isupper(): #new entry   
            if e: 
                if not c: c="1"
                comps.append([e,c])
            e,c=i,""         
        elif i.islower(): e+=i
        elif i.isdigit(): c+=i
    if e: 
        if not c: c="1"
        comps.append([e,c])
    
    cdf=pd.DataFrame(comps,columns=["elements","counts"]).set_index("elements").T.astype(int)
    cdf["+"]=form.count("+")
    cdf["-"]=form.count("-")
    return cdf

def getMz(form): #this could be vectorized for speed up in the future
    cdf=parse_form(form)
    return (cdf*mdf.loc[cdf.columns].T.values).sum().sum()

#%%

import operator
from functools import reduce
from numpy import argwhere, array, uint8,uint64,vstack

max_count="H[200]C[75]N[50]O[50]P[10]S[10]"
max_mas=""

max_df=pd.DataFrame(array(max_count.replace("]","[").split("["))[:-1].reshape(-1,2),columns=["element","max_count"])
max_df["max_count"]=max_df["max_count"].astype(uint64)+1


n = reduce(operator.mul, max_df.max_count.values, 1)

startvals=np.hstack([1, [reduce(operator.mul, max_df.max_count.values[:i+1]) for i in range(len(max_df))]]).astype(uint64)
mvals=max_df.max_count.values.astype(uint64)-1


def hash_int(a):
    #a=array(a,dtype=uint64)
    nz=argwhere(a)[:,0]
    return (startvals[nz]*a[nz]).sum()

def unhash_int(ha):
    ov=argwhere(ha>=startvals)[:,0][::-1]
    a=array([0,0,0,0,0,0])
    for o in ov:
        t=ha//startvals[o]
        ha=ha%startvals[o]
        a[o]=t 
    return a



#%%


#def unhash_int (test)
a=array([0, 0, 2, 2, 0, 0])
ha=hash_int(a)
ha=hash_int(a)



mass = getMz("CO2")*20

mass_blowup=40000
ppm=5
elements = max_df.element.tolist() #["C","H","N","O","P","S"]

element_masses=np.round(mdf.loc[elements].values*mass_blowup,0).astype(int)[:,0]
l,u=int(round(mass*(1-ppm/1e6)*mass_blowup,0)),int(round(mass*(1+ppm/1e6)*mass_blowup,0))
mi=np.arange(l,u+1)
amount=mi.max()

#%% unbounded coin change (list)


# s=time.time()
# dp = [[] for _ in range(amount + 1)]
# dp[0] = [[]]  # Base case: one way to make 0, with empty combination

# #test
# mo=[] #splitting of solutions in dpi
# mb=[] #splitting of solutions in dpi-e

# for ix,e in enumerate(element_masses):
#     for i in range(e, amount + 1):
#         for combo in dp[i - e]:
            

            
#             dp[i].append(combo + [uint8(ix)])
            
#             #test
#             if len(dp[i-e])>1: mb.append([ix,i])
#             if len(dp[i])>1:   mo.append([ix,i])

# sols=[[mi[ix], dp[i]] for ix,i in enumerate(mi) if len(dp[i])]
# elapsed=time.time()-s
# ms=[[ix,i] for ix,i in enumerate(dp) if len(i)]
# print(len(ms))
# print(elapsed)


#%% unbounded coin change array

# s=time.time()

# u=array([0,0,0,0,0,0],dtype=uint64).reshape(1,-1) 
# dp = [[] for _ in range(amount + 1)]
# dp[0] = [u]  # Base case: one way to make 0, with empty combination


# for ix,e in enumerate(element_masses):
#     for i in range(e, amount + 1):
        
        
#         if len(dp[i-e]): 
#             a=vstack(dp[i-e])
#             a[:,ix]+=1
#             if len(dp[i]): dp[i]=vstack([dp[i],a])
#             else:          dp[i]=a
            
             
# sols=[[mi[ix], dp[i]] for ix,i in enumerate(mi) if len(dp[i])]
# elapsed=time.time()-s
# ms=[[ix,i] for ix,i in enumerate(dp) if len(i)]
# print(len(ms))
# print(elapsed)





#%% unbounded coin change hased array

s=time.time()


dp = [[] for _ in range(amount + 1)]
dp[0] = [0]  # Base case: one way to make 0, with empty combination


for ix,e in enumerate(element_masses):
    for i in range(e, amount + 1):
        
        if len(dp[i-e]): 

            #unhash, hash
            a=vstack([unhash_int(c) for c in dp[i-e]])                
            a[:,ix]+=1
            a=a[a[:,ix]<=mvals[ix]] #filter on compositional bounds
            a=[uint64(hash_int(c)) for c in a] 
            
        
            if len(dp[i]): dp[i]+=a
            else:          dp[i]=a
            
             
sols=[[mi[ix], dp[i]] for ix,i in enumerate(mi) if len(dp[i])]
elapsed=time.time()-s
ms=[[ix,i] for ix,i in enumerate(dp) if len(i)]
print(len(ms))
print(elapsed)

#%% DP with int accurate bin calculation?

#%% DP with roundoff mitigation
