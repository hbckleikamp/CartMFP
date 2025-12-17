# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 14:05:02 2025

@author: e_kle
"""

import pandas as pd
freqs =  { "A":8.25, "Q":3.93, "S":6.65, "R":5.52, "E":6.71, "K":5.79, "T":5.36,
                 "N":4.06, "G":7.07, "M":2.41, "W":1.10, "D":5.46, "H":2.27, "F":3.86,   
                 "Y":2.92, "C":1.38, "P":4.74, "V":6.85, "L":16.54}  #equate IL: "L":9.64  + #"I":5.90,



elements=["H","C","N","O","P","S","Se"]
aa_molecular_formulas=pd.DataFrame([
#AA  H    C O N P S Se 
#["", 0,   0,0,0,0,0,0],
["A",5,   3,1,1,0,0,0],
#["C",5,   3,1,1,0,1,0],
["C",7,5,2,2,0,1,0], #carbamidomethylated cysteine "H(3) C(2) N O"
["D",5,   4,3,1,0,0,0],
["E",7,   5,3,1,0,0,0],
["F",9,   9,1,1,0,0,0],
["G",3,   2,1,1,0,0,0],
["H",7,   6,1,3,0,0,0],
#["I",12,  6,1,1,0,0,0],
#["J",11,  6,1,1,0,0,0],
["K",12,  6,1,2,0,0,0],
["L",11,  6,1,1,0,0,0],
["M",9,   5,1,1,0,1,0],
["N",6,   4,2,2,0,0,0],
["P",7,   5,1,1,0,0,0],
["Q",8,   5,2,2,0,0,0],
["R",12,  6,1,4,0,0,0],
["S",5,   3,2,1,0,0,0],
["T",7,   4,2,1,0,0,0],
["V",9,   5,1,1,0,0,0],
["W",10, 11,1,2,0,0,0],
["Y",9,   9,2,1,0,0,0],
#["U",5,   3,1,1,0,0,1],
#["O",19, 12,2,3,0,0,0], #no side chain neutral losses are described for U and O in expert system
#modified residues
# ["Mox", 9,5,2,1,0,1,0], #oxidized methionine
# ["Ccam",8,5,2,2,0,1,0], #carbamidomethylated cysteine "H(3) C(2) N O"
# ["Sph", 6,3,5,1,1,0,0], #phosphorylation "H O(3) P"
# ["Tph", 8,4,5,1,1,0,0],
# ["Yph",10,9,5,1,1,0,0]],
],
    columns=["AA"]+elements).set_index("AA")

aa_molecular_formulas=aa_molecular_formulas.loc[freqs.keys()]
#%% HK
import numpy as np
import pandas as pd
rng = np.random.default_rng(0)
elements=["C", "H", "N", "O", "S"]
aa_molecular_formulas=aa_molecular_formulas[elements]

p = np.array([i for i in freqs.values()])/100

confidence=0.999
confidences=[1-confidence,confidence]

n=int(1e7)

crats=[]
counts=[]
rdbes=[]

for L in np.arange(0,50):
    
    
    print(L)

    if  L: v = pd.DataFrame(rng.multinomial(L, p, size=n).dot(aa_molecular_formulas.values),columns=elements)
    else:  v = pd.DataFrame(np.zeros([n,len(elements)])                                    ,columns=elements)
        
    
    v[["H","O"]]+=[2,1] #add water

    #add here K/R? or leave it out?
    #random sample : (one fraction + K, one fraction + R)
    aK=v.sample(int(freqs["K"]/(freqs["K"]+freqs["R"])*1e6)).index #add K
    v.loc[ v.index.isin(aK),:]+=aa_molecular_formulas.loc["K",:].values
    v.loc[~v.index.isin(aK),:]+=aa_molecular_formulas.loc["R",:].values

    Cm=v["C"].mean()

    for e in ["H","N","O","S"]:
        crats.append(np.hstack([e,(v[e]/v["C"]).quantile(confidences).values,L,Cm]))
    
    #quantile 99.9 counts
    c=v.quantile(confidences)
    c["Cs"]=Cm
    c["length"]=L
    counts.append(c)
    
    #RDBE = X -Y/2+Z/2+1 where X=C Y=H or halogen, Z=N or P https://fiehnlab.ucdavis.edu/projects/seven-golden-rules/ring-double-bonds
    #quantile 99.9 rdbe
    r=pd.DataFrame([np.quantile((1+v["C"]-v["H"]/2+v["N"]/2),confidences)],columns=["min","max"])    
    r["Cs"]=Cm
    r["length"]=L
    rdbes.append(r)
    
res=pd.DataFrame(np.vstack(crats),columns=["ratio","low","high","length","Cs"])
res[["low","high","length","Cs"]]=res[["low","high","length","Cs"]].astype(float)

#%%
counts=pd.concat(counts)
rdbes=pd.concat(rdbes)

#%%

counts.to_csv("C-centric_counts.csv")
rdbes.to_csv("C-centric_rdbes.csv")
res.to_csv("C-centric_ratios.csv")




# #%%
# rl=res.pivot(index="Cs",columns="ratio",values="low")
# rh=res.pivot(index="Cs",columns="ratio",values="high")


# #%%
# rh.to_csv("peptide_Cratios_high.csv")
# rl.to_csv("peptide_Cratios_low.csv")

# #just need to interpolate


# #%%

# import matplotlib.pyplot as plt

# fig,ax=plt.subplots()
# plt.plot(rdbes["Cs"],rdbes["min"].values)
# plt.plot(rdbes["Cs"],rdbes["max"].values)

# fig,ax=plt.subplots()
# plt.plot(rl.H)
# plt.plot(rh.H)

# fig,ax=plt.subplots()
# plt.plot(rl.N)
# plt.plot(rh.N)

# fig,ax=plt.subplots()
# plt.plot(rl.O)
# plt.plot(rh.O)

# fig,ax=plt.subplots()
# plt.plot(rl.S)
# plt.plot(rh.S)
