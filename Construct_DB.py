# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:48:02 2025

@author: e_kle
"""


#%% Modules

from inspect import getsourcefile
import warnings
from functools import reduce
import operator
import sys
import psutil
from pathlib import Path
import os
import pandas as pd
import numpy as np
from collections import Counter

from npy_append_array import NpyAppendArray
from numpy.lib.format import open_memmap
from contextlib import ExitStack


# %% change directory to script directory (should work on windows and mac)

basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())
base_vars=list(locals().copy().keys()) #base variables

# %% Arguments

#composition arguments (Default)
composition="H[200]C[75]N[50]O[50]P[10]S[10]"   # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]
max_mass = 1000         # default 1000
min_rdbe = -5           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
max_rdbe = 80

#advanced chemical rules
filt_7gr="Common"                                                                   #False,True,"Common","Extended",Toggles all advanced chemical filtering using rules #2,4,5,6 of Fiehn's "7 Golden Rules" 
filt_LewisSenior=True                                                               #Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence)
filt_ratios="HC[0,6]FC[0,6]ClC[0,2]BrC[0,2]IC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]"    #Golden Rules #4,5: Filter on chemical ratios with extended range 99.9% coverage
filt_NOPS=True                                                                      #Golden Rules #6:   Filter on NOPS probabilities

#additional rules
filt_multimetal=1                                                                   #only allow x different metallic atom types
filt_halogens  =2                                                                   #only allow x different halogen atom types

#C-centric filtering files (instead of 7gr, more specific C-centric rules are used)
Crats=  ""
Crdbes= ""

#performance arguments
maxmem = 10e9 #0.7      # fraction of max free memory usage 
mass_blowup = 100000     # converting mass to int (higher blowup -> better precision, but larger indexing table)
write_mass=False         # also writes float array of masses

#filepaths
mass_table = str(Path(basedir, "mass_table.tsv"))           # table containing element masses, default: CartMFP folder/ mass_table.tsv"
Output_folder = str(Path(basedir, "CartMFP_test")) # default: CartMFP folder / Cart_Output
Output_file=""  

write_params=True #write an output file with arguments used to construct the db
sparse_index=True #store index file as sparse matrix
remove=True #removes unsorted composition file and sorted index file after building sorted array
debug=False #True


#%% store parameters
params={}
[params.update({k:v}) for k,v in locals().copy().items() if k not in base_vars and k[0]!="_" and k not in ["base_vars","params"]]
#%% Arguments for execution from command line.

if not hasattr(sys,'ps1'): #checks if code is executed from command line
    
    import argparse

    parser = argparse.ArgumentParser(
                        prog='CartMFP-space2cart',
                        description='molecular formula prediction, see: https://github.com/hbckleikamp/CartMFP')
    
    #output and utility filepaths
    parser.add_argument("-mass_table",                             default=str(Path(basedir, "mass_table.tsv")), required = False, help="list of element masses")  
    parser.add_argument("-Output_folder",  default=str(Path(basedir, "CartMFP_Database")),   required = False, help="Output folder for cartesian files")   
    parser.add_argument("-Output_file",   default="", required = False, help="Output file name for cartesian files")   
   
    #composition constraints
    parser.add_argument("-c", "--composition", default="H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]", 
    required = False, help="ALlowed elements and their minimum and maximum count. The following syntax is used: Element_name[minimum_count,maximum_count]")  
    parser.add_argument("-max_mass",  default=1000, required = False, help="maximum mass of compositions",type=float)  
    parser.add_argument("-min_rdbe",  default=-5,   required = False, help="minimum RDBE of compositions. set False to turn off",type=float)  
    parser.add_argument("-max_rdbe",  default=80,   required = False, help="maximum RBDE of compositions. set False to turn off",type=float)  

    #advanced composition constraints
    parser.add_argument("-filt_7gr",  default=True,   required = False, help="Toggles all advanced chemical filtering using rules #2,4,5,6 of Fiehn's 7 Golden Rules ")
    parser.add_argument("-filt_LewisSenior",  default=True,   required = False, help="Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence)")
    parser.add_argument("-filt_ratios",  default="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]IC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]" ,   required = False, help="Golden Rules #4,5: Filter on chemical ratios with extended range 99.9% coverage")  
    parser.add_argument("-filt_NOPS",  default=True,   required = False, help="Golden Rules #6:   Filter on NOPS probabilities")
    parser.add_argument("-filt_multimetal",  default=1,   required = False, help="Set maximum for the number of different types of metal atoms in one prediction")
    parser.add_argument("-filt_halogens",  default=2,   required = False, help="Set maximum for the number of different halogens in one prediction")

    #C-centric ratios (Optional)
    parser.add_argument("-Crats",   default="",  required = False, help="Toggles all advanced chemical filtering using rules #2,4,5,6 of Fiehn's 7 Golden Rules ")
    parser.add_argument("-Crdbes",  default="",   required = False, help="Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence)")

    #performance arguments
    parser.add_argument("-mem",  default=0.7, required = False, help="if <=1: max fraction of available RAM used, if >1: mass RAM usage in GB",type=float)  
    parser.add_argument("-mass_blowup",  default=100000, required = False, help="multiplication factor to make masses integer. Larger values increase RAM usage but reduce round-off errors",type=int)  
    parser.add_argument("-write_mass",  default=False, required = False, help="Also create a lookup table for float masses",type=int)  
    parser.add_argument("-sparse_index ",  default=True, required = False, help="Reduces index table size, at the cost of annotation speed",type=int)  
        
    
    #write params
    parser.add_argument("-write_params",  default=True, required = False, help="writes parameter file")  
    parser.add_argument("-remove",  default=True, required = False, help="removes unsorted composition file and sorted index file after building sorted array")  
    parser.add_argument("-d","--debug",  default=False, required = False, help="")  
    

    args = parser.parse_args()
    params = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    
    print("")
    print(args) 
    print("")
    locals().update(params)


# %% General functions

# Source: Eli Korvigo (https://stackoverflow.com/questions/28684492/numpy-equivalent-of-itertools-product/28684982)
#https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
#https://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
#from sklearn.utils.extmath import cartesian
def cartesian(arrays, bitlim=np.uint8, out=None):
    n = reduce(operator.mul, [x.size for x in arrays], 1)
    print(n)
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=bitlim)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if len(arrays[1:]): #arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


from itertools import combinations, product
def limited_cartesian(vectors, k):
    results = []
    for idxs in combinations(range(len(vectors)), k):
        vecs = [vectors[i] for i in idxs]
        for values in product(*vecs):
            results.append((idxs, values))
    return results


# read input table (dynamic delimiter detection)
def read_table(tabfile, *,
               Keyword=[], #rewrite multiple keywords,
               xls=False,
               dlim=""
               ):
    if type(Keyword)==str: Keyword=[i.strip() for i in Keyword.split(",")]
    
    #numpy format
    if tabfile.endswith(".npy"): 
        tab=np.load(tabfile)
        if len(Keyword): tab=pd.DataFrame(tab,columns=Keyword)
        return True,tab
    
    
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        if len(dlim):
            try:
                tab=pd.read_csv(tabfile,sep=dlim)
                return True,tab
            except:
                pass

        #try opening with xlsx
        if tabfile.endswith(".xls") or tabfile.endswith(".xlsx") or xls:
            try:
                tab = pd.read_excel(tabfile, engine='openpyxl')
                return True,tab
            except:
                pass
        
        # dynamic delimiter detection: if file delimiter is different, split using different delimiters until the desired column name is found
        if len(Keyword):
            
            with open(tabfile, "r") as f:
                tab = pd.DataFrame(f.read().splitlines())
            
            
            if not tab.columns.isin(Keyword).any():
                delims = [i[0] for i in Counter(
                    [i for i in str(tab.iloc[0]) if not i.isalnum()]).most_common()]
                for delim in delims:
                    if delim == " ":
                        delim = "\s"
                    try:
                        tab = pd.read_csv(tabfile, sep=delim)
                        if tab.columns.isin(Keyword).any():
                            return True,tab
                    except:
                        pass

    return False,tab



#The number of bits needed to represent an integer n is given by rounding down log2(n) and then adding 1
def bits(x,neg=False):
    bitsize=np.array([8,16,32,64])
    dtypes=[np.uint8,np.uint16,np.uint32,np.uint64]
    if neg: dtypes=[np.int8,np.int16,np.int32,np.int64]
    return dtypes[np.argwhere(bitsize-(np.log2(x)+1)>0).min()]

#https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
def create_ranges(a):
    l = a[:,1] - a[:,0]
    clens = l.cumsum()
    ids = np.ones(clens[-1],dtype=int)
    ids[0] = a[0,0]
    ids[clens[:-1]] = a[1:,0] - a[:-1,1]+1
    return ids.cumsum()

#vectorized find nearest mass
#https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
def find_closest(A, target): #returns index of closest array of A within target
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

#test
def cm(comps):
 return (np.sum(comps*mdf.loc[edf.index].values.T,axis=1)*mass_blowup).round(0).astype(int)

# %% Get elemental metadata 


if os.path.exists(mass_table): mdf = pd.read_csv(mass_table, index_col=[0], sep="\t")
else:                         raise ValueError("Mass table "+mass_table+" not found!")
mdf,vdf=mdf["mass"],mdf["Valence"]-2

emass = 0.000548579909  # electron mass
mdf.loc["+"]=-emass
mdf.loc["-"]=+emass

halogens=["Br","Cl","F","I"]
metals=["Na","Mg","Al","Si","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se",
        "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","Cs","Ba","la","Ce",
        "Pr","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb"]

#print warnings if files not found
if (len(Crdbes)) & (not os.path.exists(Crdbes)): print("Warning!: C-centric rdbe file not found! ")
if (len(Crats)) & (not os.path.exists(Crats)): print("Warning!: C-centric ratio file not found! ")
    
# %% Construct MFP space


# % Construct elemental space dataframe
edf=pd.DataFrame([i.replace(",","[").split("[") if "," in i else [i.split("[")[0],0,i.split("[")[-1]] for i in composition.split("]")[:-1]] ,columns=["symbol","low","high"]).set_index("symbol")

edf["low"]=pd.to_numeric(edf["low"],errors='coerce')
edf["high"]=pd.to_numeric(edf["high"],errors='coerce')
edf=edf.ffill(axis=1)


if filt_7gr=="Common": #max element count restriction (Wiley/DNP)
    crep=pd.DataFrame([[ 39, 72,20,20,9,10,16,10, 4, 4,  8],
                       [ 78,126,25,27,9,14,34,12, 8, 8, 14],
                       [156,236,32,63,9,14,48,12,10,10, 15],
                       [162,236,48,78,9,14,48,12,10,10, 15]],
                      index=[500,1000,2000,3000],
                      columns=["C","H","N","O","P","S","F","Cl","Br","I","Si"])

    #add missing elements
    # mel=list(set(edf.index)-set(crep.columns))
    # crep[mel]=edf.high[mel].values

    # for n,r in crep.iterrows():
    #     if n>=max_mass: 
    #         m=edf.merge(r,left_index=True,right_index=True,how="left")
            #edf.loc[m.index,"high"]=m[n].values             #this overwrites compositional space values!
            
            #instead overwrite crep with minimum 
            # mrep=m[["high",n]].min(axis=1)        
            # crep.loc[n,mrep.index]=mrep.values
            
if edf.isnull().sum().sum(): #fill in missing values from composotion string.
    print("Warning! missing element maxima detected in composition. Imputing from maximum mass (this might affect performance)")
    edf.loc[edf["high"].isnull(),"high"]=(max_mass/mdf.loc[edf.index]).astype(int).values[edf["high"].isnull()].flatten()


edf[["low","high"]]=edf[["low","high"]].fillna(0).astype(int)
edf["high"]=np.clip(edf["high"],0,(max_mass/mdf.loc[edf.index]).astype(int)) #limit to max mass
edf = edf.sort_values(by="high", ascending=False)


if filt_halogens: #put halogens last
    q=edf.index.isin(halogens)
    edf=pd.concat([edf[~q],edf[q]]) 

if filt_multimetal: #put metals last
    q=edf.index.isin(metals)
    edf=pd.concat([edf[~q],edf[q]]) 

#put C at zero for C-centric filtering
edf=pd.concat([edf[edf.index=="C"],edf[edf.index!="C"]])
q=edf.index.isin(["C","H","N","P"]) #test for more efficient rdbe calculation
edf=pd.concat([edf[q],edf[~q]])


edf["arr"] = edf.apply(lambda x: np.arange(
    x.loc["low"], x.loc["high"]+1), axis=1)
edf["fmass"] = mdf.loc[edf.index] #float mass
edf["mass"]  = (edf["fmass"]*mass_blowup).astype(np.uint64)
elements=edf.index.values
params["elements"]=elements.tolist()
metcols=np.hstack([np.argwhere(elements==m)[:,0] for m in metals])
halcols=np.hstack([np.argwhere(elements==m)[:,0] for m in halogens])


if filt_7gr=="Common": #determine column order for Common max element filtering
    mc=edf.index.isin(crep.columns)   
    mcx=np.argwhere(mc)[:,0]
    crep=crep[edf.index[mc]]
    crep.index*=mass_blowup
 


bitlim=np.uint8
if edf.high.max()>255: 
    bitlim=np.uint16
    print("element cound above 255 detected, using 16bit compositions")
    print("")

bmax = int(np.round((max_mass+1)*mass_blowup, 0))

# % Determine number of element batches
mm = psutil.virtual_memory()
mem0=mm.used
dpoints = np.array([10, 100, 1e3, 1e4, 1e5, 1e6]).astype(int)

# size of uint8 array
onesm = np.array([sys.getsizeof((np.ones(i)).astype(bitlim)) for i in dpoints])
a8, b8 = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

# size of uint64 array
onesm = np.array([sys.getsizeof((np.ones(i)).astype(np.uint64)) for i in dpoints])
a64, b64 = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

#size of float array
onesm = np.array([sys.getsizeof((np.ones(i)).astype(np.float64)) for i in dpoints])
afloat, bfloat = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

#this fitting section is a bit meaningless, since its just 2**8 for 64 bit ->8

rows = 1
memories = []
cols = len(edf)
for i in edf.arr:
    rows *= len(i)
    s_uint8 = rows*cols*a8[0]+b8[0]
    s_uint64 = rows*a64[0]+b64[0] 
    memories.append(s_uint8+s_uint64)



if maxmem>mm.total:   
    print("Warning: supplied memory usage larger than total RAM, lowering to 50% of total RAM")
    maxmem=0.5 

if maxmem<1: 
    maxRam=mm.free*maxmem
    maxPercent=maxmem*100
else:        
    maxRam=maxmem
    maxPercent=(1-maxRam/mm.total)*100

mem_cols = (np.argwhere(np.array(memories) < (maxRam))[-1]+1)[0]
need_batches = len(edf)-mem_cols

#%% Parse chemical rules

if type(filt_7gr)==str:
    if filt_7gr=="False" or filt_7gr=="0" or filt_7gr=="": filt_7gr=False


if filt_7gr=="Common":
    filt_LewisSenior=True
    filt_NOPS=True
    filt_ratios="HC[0.2,3.1]FC[0,1.5]ClC[0,0.8]BrC[0,0.8]IC[0,0.8]NC[0,1.3]OC[0,1.2]PC[0,0.3]SC[0,0.8]SiC[0,0.5]" #common ratio range
    #min_rdbe=0
    
if filt_7gr=="Extended":
    filt_LewisSenior=True
    filt_NOPS=True
    filt_ratios="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]IC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]" #extended ratio range

if not filt_7gr or filt_7gr=="False": #turn of 7gr [except for custom filt_ratios]
    print("No 7gr filtering done!")
    filt_LewisSenior=False
    filt_NOPS=False
    filt_ratios=False #if filt_ratios=="HC[0,6]FC[0,6]ClC[0,2]BrC[0,2]IC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]": 

if not "[" in str(filt_ratios): filt_ratios=False
    

if write_params: #update params
    params["filt_7gr"]=filt_7gr
    params["filt_LewisSenior"]=filt_LewisSenior
    params["filt_NOPS"]=filt_NOPS
    params["filt_ratios"]=filt_ratios
    

#parse chemical ratios [Golden rules #4,5]
erats,batch_rats=[],[]
if filt_ratios:
    
    sf=filt_ratios.replace("]","[").split("[")[:-1]
    erats=pd.DataFrame([i.split(",") for i in sf[1::2]],columns=["low","high"])  

    efilts=[]
    for i in sf[::2]:
        x=np.argwhere([s.isupper() for s in i])[1][0]
        efilts.append([i[:x],i[x:]])
    erats[["l","r"]]=efilts
    
    erats=erats[(erats.l.isin(elements)) & (erats.r.isin(elements))]
    if len(erats):
        erats["lix"]=[np.argwhere(elements==e)[0,0] for e in erats["l"]]
        erats["rix"]=[np.argwhere(elements==e)[0,0] for e in erats["r"]]
    
        #fill missing values
        erats=erats.fillna("0")
        q=(erats["high"]==0) | (erats["high"]=="")
        erats.loc[q,"high"]=edf.iloc[erats.loc[q,"lix"]]["high"].values
        erats[["low","high"]]=erats[["low","high"]].astype(float)
        
      
       


#%% construct output paths

if Output_file=="":
    Output_file = "".join(edf.index+"["+edf.low.astype(str)+","+edf.high.astype(
        str)+"]")+"_b"+str(mass_blowup)+"max"+str(int(max_mass))+"rdbe"+str(min_rdbe)+"_"+str(max_rdbe) 
    if filt_7gr: Output_file+="_7gr"
    if filt_7gr=="Common":      Output_file+="Common"
    elif filt_7gr=="Extended":  Output_file+="Extended"
    elif filt_ratios!="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]": Output_file+="_customfilt"
    Output_file=Output_file.replace("[0,","[")
else: 
    write_params=True



if not len(Output_folder):
    Output_folder = os.getcwd()
else:
    if not os.path.exists(Output_folder):
        os.makedirs(Output_folder)

basepath=str( Path(Output_folder, Output_file))
unsorted_comp_output_path =basepath+"_unsorted_comp.npy"
mass_output_path =         basepath+"_mass.npy"
comp_output_path =         basepath+"_comp.npy"
m2g_output_path  =         basepath+"_m2g.npy"



print("Output Cartesian file:")
print(Output_file)
print("")


# Chemical filtering
flag_rdbe_min = type(min_rdbe) == float or type(min_rdbe) == int
flag_rdbe_max = type(max_rdbe) == float or type(max_rdbe) == int
flag_crdbe    = os.path.exists(Crdbes) #C-centric rdbe 
flag_crats    = os.path.exists(Crats) #C-centric element ratios




if flag_rdbe_min or flag_rdbe_max or flag_crdbe:
    # #chemical filtering for RDBE (it is int rounded so pick a generous range)
    # #RDBE = X -Y/2+Z/2+1 where X=C Y=H or halogen, Z=N or P https://fiehnlab.ucdavis.edu/projects/seven-golden-rules/ring-double-bonds
    Xrdbe = np.argwhere(edf.index == "C").flatten()
    Yrdbe = np.argwhere(edf.index.isin(["H", "F", "Cl", "Br", "I"])).flatten()
    Zrdbe = np.argwhere(edf.index.isin(["N", "P"])).flatten()
    rdbe_bitlim=np.int16



# % Compute base cartesian
print("constructing base cartesian:")
arrays = edf.arr.values[:mem_cols].tolist()


zm = cartesian(arrays,bitlim=bitlim)
mass = np.zeros(len(zm), dtype=np.uint64)

#memory efficient batched addition 
remRam=maxRam-(mm.free-psutil.virtual_memory().free)
stepsize=np.round(remRam/(afloat*mem_cols)/2,0).astype(int)
ixs=np.arange(0,len(zm)+stepsize,stepsize)
          
for i in range(len(ixs)-1):
    mass[ixs[i]:ixs[i+1]]=((zm[ixs[i]:ixs[i+1]]*mdf.loc[edf.index].values[:mem_cols].T).sum(axis=1)*mass_blowup).round(0).astype(np.uint64)

#%%


# filter base cartesian on maximum mass
if max_mass:  
    zm = zm[mass <= max_mass*mass_blowup]
    mass = mass[mass <= max_mass*mass_blowup]

#prefilter on elemental ratios [Golden rules #4,5]
if len(erats):
    pre_els=np.array(edf.index[:mem_cols])
    prerats=erats[(erats[["l","r"]].isin(pre_els)).all(axis=1)]
    
    if len(prerats):
        q=(prerats["lix"]<mem_cols) &  (prerats["rix"]<mem_cols)    
        base_rats,batch_rats=prerats[q],prerats[~q]
        
        #prefilter on chemical ratios
        q=np.ones(len(mass),bool)
        for _,rat in base_rats.iterrows():
            r=zm[:,rat.lix]/zm[:,rat.rix]
            q=q & ((~np.isfinite(r)) | ((r>=rat.low) & (r<=rat.high)))
        mass,zm=mass[q],zm[q]

#%% C-centric filtering


#### Parse ratio files ####
if flag_crats:
    res=pd.read_csv(Crats) #add conditions for file detection
    ratl=res.pivot(index="C",columns="ratio",values="low")
    rath=res.pivot(index="C",columns="ratio",values="high")
    
    countsl=ratl*ratl.index.values.reshape(-1,1)
    countsh=rath*rath.index.values.reshape(-1,1)
    
    #linreg missing values
    if edf.high.C>countsl.index.max():
        x=countsl.index[-10:] #linreg on last 10 values
        xf=np.arange(countsl.index.max(),edf.high.C+1)
        fs=[]
        for c in countsl.columns:
            y=countsl[c].values[-10:]
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y)[0]
            fs.append(xf*a+b)
        countsl=pd.concat([countsl,pd.DataFrame(np.vstack(fs).T,columns=countsl.columns,index=xf)])
    
    #interpolate as ints
    rs=np.arange(countsl.index.min(),countsl.index.max()).astype(int)
    countsl=pd.DataFrame(np.vstack([np.interp(rs,countsl.index,countsl[c]).astype(int) for c in countsl.columns]).T,columns=countsl.columns,index=rs)
    
    #linreg missing values
    if edf.high.C>countsh.index.max():
        x=countsh.index[-10:] #linreg on last 10 values
        xf=np.arange(countsh.index.max(),edf.high.C+1)
        fs=[]
        for c in countsh.columns:
            y=countsh[c].values[-10:]
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y)[0]
            fs.append(xf*a+b)
        countsh=pd.concat([countsh,pd.DataFrame(np.vstack(fs).T,columns=countsh.columns,index=xf)])
    
    #interpolate as ints
    rs=np.arange(countsh.index.min(),countsh.index.max()+1).astype(int)
    countsh=pd.DataFrame(np.vstack([np.interp(rs,countsh.index,countsh[c]).astype(int) for c in countsh.columns]).T,columns=countsh.columns,index=rs)
    
    #Future: add missing elements (easier indexing)

#### Parse rdbe files ####
if flag_crdbe:

    Crdbes=pd.read_csv(Crdbes,index_col=[0]) #add conditions for file detection

    #regress missing values
    if edf.high.C>Crdbes.index.max():
        x=Crdbes.index[-10:] #linreg on last 10 values
        xf=np.arange(Crdbes.index.max(),edf.high.C+1)
        fs=[]
        for c in Crdbes.columns:
            y=Crdbes[c].values[-10:]
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y)[0]
            fs.append(xf*a+b)
        Crdbes=pd.concat([Crdbes,pd.DataFrame(np.vstack(fs).T,columns=Crdbes.columns,index=xf)]).astype(int)

    #interpolate 
    rs=np.arange(Crdbes.index.min(),Crdbes.index.max()+1).astype(int)
    Crdbes=pd.DataFrame(np.vstack([np.interp(rs,Crdbes.index,Crdbes[c]).astype(int) for c in Crdbes.columns]).T,columns=Crdbes.columns,index=rs)
    Crdbes=(Crdbes*2).astype(int) #blowup to int



#%% prefiltering

if flag_crats:

    #filter minimum element counts
    q=(zm[:,1:mem_cols]>=countsl[edf.index[1:mem_cols]].values[zm[:,0]-countsl.index[0]]).all(axis=1)
    zm,mass=zm[q],mass[q]
    
    #filter maximum element counts
    q=(zm[:,1:mem_cols]<=countsh[edf.index[1:mem_cols]].values[zm[:,0]-countsh.index[0]]).all(axis=1)
    zm,mass=zm[q],mass[q]


if flag_rdbe_min or flag_rdbe_max or flag_crdbe:

    rdbe_cols=np.hstack([Xrdbe,Yrdbe,Zrdbe])
    if (rdbe_cols<mem_cols).all(): #check rdbe columns
        
        base_rdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
        base_rdbe +=(zm[:, Xrdbe].sum(axis=1)*2).astype(rdbe_bitlim)
        base_rdbe -=zm[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
        base_rdbe +=zm[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim)
        
        q=np.ones(len(mass),bool)
        
        #prefilter on C-centric rdbe values
        if flag_crdbe:
            q= q & (base_rdbe>=Crdbes.values[zm[:,0]-Crdbes.index[0],0]) & (base_rdbe<=Crdbes.values[zm[:,0]-Crdbes.index[0],1])
   
        
        #prefilter on global rdbe values
        if flag_rdbe_min or flag_rdbe_max:
            
            if flag_rdbe_min: q=q & (base_rdbe>=min_rdbe)
            if flag_rdbe_max: q=q & (base_rdbe<=max_rdbe)
        
        mass,zm=mass[q],zm[q]
        
        #turn of rdbe check, since it is done
        flag_rdbe_max,flag_rdbe_min,flag_crdbe=False,False,False
    
#parse NOPS probability [Golden rule #6]
nops=[]
if filt_NOPS:  #this can be modified with a custom DF if needed
    nops=pd.DataFrame([ [["N","O","P","S"],	1,	10,	20,	4,	3],
                        [["N","O","P"],	    3,	11,	22,	6,	0],
                        [["O","P","S"],	    1,	0,	14,	3,	3],
                        [["P","S","N"],	    1,	3,	3,	4,	0],
                        [["N","O","S"],	    6,	19,	14,	0,	8]],
                      columns=["els","lim","N","O","P","S"])
    
    #remove rows that are not in elements
    nops=nops[[np.all(np.in1d(np.array(i),elements)) for i in nops.els]]
    
    if len(nops):
        #replace 0 with max counts
        for e in nops.columns[2:]:
            if e in elements: nops.loc[nops[e]==0,e]=edf.loc[e,"high"] #fill missing values
            else:             nops.pop(e)
        nops["ixs"]=[np.array([np.argwhere(e==elements)[0][0] for e in i]) for i in nops.els.values]
            
#precompute NOPS probability [Golden rule #6]
if len(nops):
    # nops_ixs=np.hstack([np.argwhere(elements==e).flatten() for e in nops.columns[2:]])
    # nops_vals=nops[nops.columns[2:]].values
    nops["mx"]=nops.ixs.apply(max)
    
    #prefilter on NOPS probabilities (Slow)
    q=np.ones(len(mass),bool)
    for x,row in nops.iterrows():
        if row.mx<zm.shape[1]:   
            #q=q & (((zm[:,row.ixs]==0).any(axis=1)) | (np.all(zm[:,row.ixs]<=row[row.els].values,axis=1)))
            #q=q & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,nops_ixs]<row.values[2:-1],axis=1)))  #OR above lim OR 
            q=q & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,row.ixs]<=row[row.els].values,axis=1)))
    
    mass,zm=mass[q],zm[q]
    
    #nops filtering is done, no more need

    nops=nops[nops.mx>=mem_cols].reset_index(drop=True)


#%%
if zm.max()<256: bitlim=np.uint8



#%%

# add room for remaining columns
zm = np.hstack(
    [zm, np.zeros([len(zm), len(edf)-mem_cols], dtype=bitlim)])



#%% calculate base batch

if need_batches:

    #batches = reduce(operator.mul, edf.iloc[mem_cols:]["high"].values+1, 1)
    
    print("")
    print("computing remaining cartesian:")
    
    #base batches (this causes memory problems when too many elements are present)
    rows=edf[mem_cols:]
    es,arrays=rows.index,rows.arr.values
    
    #limit batches
    if filt_halogens or filt_multimetal:
        
        qhal=es.isin(halogens)
        qmet=es.isin(metals)
        nhmr=rows[~((qmet) | (qhal))]
        nhma=nhmr.arr.values
    
        barrs,bcols=[],[]
        if len(nhma):
            barrs.append(nhma)
            bcols.append(nhmr.index.values)
            
        
        if filt_halogens:   
            if qhal.sum():
                zhals=[]
                for k in np.arange(1,filt_halogens+1):
                    #mbs=limited_cartesian([i[i>0] for i in rows.arr[qhal]],k=k)
                    mbs=limited_cartesian([i for i in rows.arr[qhal]],k=k)
                    zhal=np.zeros((len(mbs),qhal.sum()),bitlim)
                    for rx,r in enumerate(mbs):
                        zhal[rx,r[0]]=r[1]
                    zhals.append(zhal)
                zhals=np.vstack(zhals)
                zhals=zhals[(zhals*edf.mass[mem_cols:][qhal].values).sum(axis=1)<max_mass*mass_blowup] #mass filter
                zhals=np.unique(zhals,axis=0)
            
                barrs.append(zhals)
                bcols.append(rows.index[qhal])
 
        
        if filt_multimetal: 
            if qmet.sum():
                zmets=[]
                for k in np.arange(1,filt_halogens+1):
                    #mbs=limited_cartesian([i[i>0] for i in rows.arr[qmet]],k=k)
                    mbs=limited_cartesian([i for i in rows.arr[qmet]],k=k)
                    zmet=np.zeros((len(mbs),qmet.sum()),bitlim)
                    for rx,r in enumerate(mbs):
                        zmet[rx,r[0]]=r[1]
                    zmets.append(zmet)
                zmets=np.vstack(zmets)
                zmets=zmets[(zmets*edf.mass[mem_cols:][qmet].values).sum(axis=1)<max_mass*mass_blowup] #mass filter
                zmets=np.unique(zmets,axis=0)
                
                barrs.append(zmets)
                bcols.append(rows.index[qmet])
        
        bm=np.vstack([np.hstack(p).astype(bitlim) for p in product(*barrs)])
        edf=pd.concat([edf[:mem_cols],edf.loc[np.hstack(bcols),:]]) #re-order columns
        bm=np.unique(bm,axis=0)
        
    else:
        bm = cartesian(arrays,bitlim=bitlim) 
    
    #filter mass
    bm=bm[(bm*edf.mass[mem_cols:].values).sum(axis=1)<max_mass*mass_blowup] #mass filter
    
    #resort edf and bm
    us=[np.array(list(set(i))) for i in bm.T]
    ls=[len(i) for i in us]
    s=np.argsort(ls)
    us,ls,bm=[us[i] for i in s],[ls[i] for i in s],bm[:,s]
    edf=pd.concat([edf[:mem_cols],edf[mem_cols:].iloc[s]])

    ##Update filters!
    
    #reset filtering indices
    elements=edf.index
    metcols=np.hstack([np.argwhere(elements==m)[:,0] for m in metals])
    halcols=np.hstack([np.argwhere(elements==m)[:,0] for m in halogens])

    #rdbe
    Xrdbe = np.argwhere(edf.index == "C").flatten()
    Yrdbe = np.argwhere(edf.index.isin(["H", "F", "Cl", "Br", "I"])).flatten()
    Zrdbe = np.argwhere(edf.index.isin(["N", "P"])).flatten()
    
    #chemical ratios
    if len(erats):
        erats["lix"]=[np.argwhere(elements==e)[0,0] for e in erats["l"]]
        erats["rix"]=[np.argwhere(elements==e)[0,0] for e in erats["r"]]
    
    #nops
    if len(nops):
        nops=nops[[np.all(np.in1d(np.array(i),elements)) for i in nops.els]]
        if len(nops):
            #replace 0 with max counts
            for e in nops.columns[2:]:
                if e in elements: nops.loc[nops[e]==0,e]=edf.loc[e,"high"] #fill missing values
                else:             nops.pop(e)
            nops["ixs"]=[np.array([np.argwhere(e==elements)[0][0] for e in i]) for i in nops.els.values]
    
        #nops filtering is done, no more need
        nops["mx"]=nops.ixs.apply(max)
        nops=nops[nops.mx>=mem_cols] 
        
    #update common element counts
    if filt_7gr=="Common": 
        mc=edf.index.isin(crep.columns)   
        mcx=np.argwhere(mc)[:,0]
        crep=crep[edf.index[mc]]
        #crep=crep[elements]
    
    #%%
     
    #precompute rdbe
    if flag_rdbe_max or flag_rdbe_min:
        
        #base rdbe
        base_rdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
        if len(Xrdbe[Xrdbe<mem_cols]): base_rdbe +=(zm[:, Xrdbe].sum(axis=1)*2).astype(rdbe_bitlim)
        if len(Yrdbe[Yrdbe<mem_cols]): base_rdbe -=zm[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
        if len(Zrdbe[Zrdbe<mem_cols]): base_rdbe +=zm[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim)

        #batch rdbe
        batch_rdbeX,batch_rdbeY,batch_rdbeZ=Xrdbe-mem_cols,Yrdbe-mem_cols,Zrdbe-mem_cols
        batch_rdbeX,batch_rdbeY,batch_rdbeZ=batch_rdbeX[batch_rdbeX>-1],batch_rdbeY[batch_rdbeY>-1],batch_rdbeZ[batch_rdbeZ>-1]
        batch_rdbe=np.zeros(len(bm),dtype=rdbe_bitlim)
        if len( batch_rdbeX): batch_rdbe +=(bm[:, batch_rdbeX].sum(axis=1)*2).astype(rdbe_bitlim)
        if len( batch_rdbeY): batch_rdbe -=bm[:, batch_rdbeY].sum(axis=1).astype(rdbe_bitlim)
        if len( batch_rdbeZ): batch_rdbe +=bm[:, batch_rdbeZ].sum(axis=1).astype(rdbe_bitlim)
        
        #prefilter on base rdbe
        q=np.ones(len(mass),bool)
        if flag_rdbe_min: q=q & ((base_rdbe+batch_rdbe.max())>=min_rdbe)
        if flag_rdbe_max: q=q & ((base_rdbe+batch_rdbe.min())<=max_rdbe)
        mass,zm=mass[q],zm[q]

    #precompute dbe (LEWIS & SENIOR rules) [Golden rule #2]
    if filt_LewisSenior: #integer dbe
        
        #do everything x2 -> faster integer calculation
        base_dbe=np.sum(zm*vdf.loc[elements].values,axis=1)+2
        batch_dbe=np.sum(bm*vdf.loc[elements[mem_cols:]].values,axis=1)

        #prefilter on dbe (remove non integer ("odd") dbe
        if not np.sum(batch_dbe%2): 
            q=(base_dbe%2)==0
            mass,zm=mass[q],zm[q]
            
    #precompute chemical ratios [Golden rules #4,5]
    if len(erats):
        q=(erats["lix"]<mem_cols) &  (erats["rix"]<mem_cols)    
        base_rats,batch_rats=erats[q],erats[~q]
        
        #prefilter on chemical ratios
        q=np.ones(len(mass),bool)
        for _,rat in base_rats.iterrows():
            r=zm[:,rat.lix]/zm[:,rat.rix]
            q=q & ((~np.isfinite(r)) | ((r>=rat.low) & (r<=rat.high)))
        mass,zm=mass[q],zm[q]
    
    #%% try to expand base
    #test=bm.copy()

    print("Expanding base cartesian")

    #initialize
    expand_col=0
    
    #while maxRam/(sys.getsizeof(mass)+sys.getsizeof(zm))>ls[expand_col]:
    while maxRam/(sys.getsizeof(zm)*10)>(ls[expand_col]*2):
    
        
        print(expand_col)
        sub_batch=us[expand_col]
        sub_batch=sub_batch[sub_batch>0]
        col=mem_cols+expand_col
        
        #get erat columns
        el=edf.index[col]
        if len(erats): rr=erats[(erats[["l","r"]]==e).any(axis=1)]
        hal=edf.index[:col+1].isin(halogens) #get halo columns
        met=edf.index[:col+1].isin(metals) #get metal columns
        
        zms,ms=[zm],[mass]
        for b in sub_batch:
            mat,matm=zm.copy(),mass.copy()
            mat[:,col]=b
            q=np.ones(len(zm),bool)

            #max mass filter
            sbm=edf.mass.values[col]*b
            matm=matm+sbm #(float mass error!)
            q=q & (matm<max_mass*mass_blowup)
    
            if bool(filt_halogens) & (sum(hal)>filt_halogens): #filt max halogens
                q=q & ((mat[:,np.argwhere(hal)[:,0]]>0).sum(axis=1)<=filt_halogens)
            if bool(filt_multimetal) & (sum(met)>filt_multimetal): #filt max metals
                q=q & ((mat[:,np.argwhere(met)[:,0]]>0).sum(axis=1)<=filt_multimetal)
        
            mat,matm=mat[q],matm[q]
            # zm,mass=np.vstack(zms),np.hstack(ms) #here?
 

            #ratio_filter
            if len(erats): #(memory leak?)
                q=np.ones(len(mat),bool)
                if len(rr): #filt elemental ratios
                    for _,rat in rr.iterrows():
                        r=mat[:,rat.lix]/mat[:,rat.rix]
                        q=q & ((~np.isfinite(r)) | ((r>=rat.low) & (r<=rat.high)))
                    mat,matm=mat[q],matm[q]
                    
           
            zms.append(mat)
            ms.append(matm)
            #ms.append(matm+sbm)
            
        zm,mass=np.vstack(zms),np.hstack(ms) 


      
        del zms,ms

        ### c-centric filtering
        if flag_crats:
            if el in countsl.columns: #expand upon input data types
                q=(zm[:,expand_col+mem_cols]>=(countsl[el].values[zm[:,0]-countsl.index[0]])) & (zm[:,expand_col+mem_cols]<=(countsh[el].values[zm[:,0]-countsh.index[0]]))
                zm,mass=zm[q],mass[q]
        
        #if all rdbe columns are present
        if (rdbe_cols<mem_cols+expand_col).all(): #check rdbe columns
            brdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
            brdbe +=(zm[:, Xrdbe].sum(axis=1)*2).astype(rdbe_bitlim)
            brdbe -=zm[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
            brdbe +=zm[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim)
        
            #filter on C-centric rdbe values
            if flag_crdbe:
                q=(brdbe>=Crdbes.values[zm[:,0]-Crdbes.index[0],0]) & (brdbe<=Crdbes.values[zm[:,0]-Crdbes.index[0],1])
                zm,mass=zm[q],mass[q]
                
            #filter on global rdbe values
            if flag_rdbe_min or flag_rdbe_max:
                q=np.ones(len(mass),bool)
                if flag_rdbe_min: q=q & (brdbe>=min_rdbe)
                if flag_rdbe_max: q=q & (brdbe<=max_rdbe)
                mass,zm=mass[q],zm[q]
    
            #turn of rdbe check, since it is done
            flag_rdbe_max,flag_rdbe_min=False,False
    
        expand_col+=1
        if expand_col>=len(ls): break
    
    #update
    mem_cols+=expand_col #update memcols
    bm=np.unique(bm[:,expand_col:],axis=0)
    if not len(bm.flatten()): need_batches=False

    #nops
    if len(nops): 
        #prefilter on NOPS probabilities (Slow)
        q=np.ones(len(mass),bool)
        for x,row in nops.iterrows():
            if row.mx<zm.shape[1]:   
                q=q & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,row.ixs]<=row[row.els].values,axis=1)))
        
        mass,zm=mass[q],zm[q]
        nops=nops[nops.mx>=mem_cols].reset_index(drop=True)

            
            
    mass=((zm*edf.fmass.values).sum(axis=1)*mass_blowup).round(0).astype(int) #recalculate float masses



#%%

s = np.argsort(mass)  #default instead of mergesort
mass, zm = mass[s], zm[s]


emp = open_memmap(m2g_output_path, mode="w+", shape=(bmax+1*mass_blowup,2),dtype=np.int64) #bits(bmax,neg=True))

if write_params: 
    import json
    
    params.update({"elements":edf.index.tolist()})             #update elements
    params.update({"column_masses":edf.fmass.values.tolist()}) #add column masses
    with open(basepath+".params", 'w') as f:
        json.dump(params, f)


#%%
if need_batches:

    #filter batches based on C-centric ratios
    if flag_crats:
        mins,maxs=[],[]
        for ix,e in enumerate(edf.index[mem_cols:]):
            d=np.vstack([countsl.index[np.clip(np.searchsorted(countsl[e],bm[:,ix]),0,len(countsl)-1)],
                         countsh.index[np.clip(np.searchsorted(countsh[e],bm[:,ix]),0,len(countsh)-1)]]).T    
            mins.append(d.min(axis=1))
            maxs.append(d.max(axis=1))
        minC,maxC=np.vstack(mins).T.max(axis=1),np.vstack(maxs).T.min(axis=1)
        q=maxC>=minC
        bm,minC,maxC=bm[q],minC[q],maxC[q]

    #add 0 batch
    zb=edf.low[mem_cols:].values
    if not np.all(bm==zb,axis=1).any():
        bm=np.vstack([zb,bm])

    #sort batches on bm
    am  = ((bm*mdf.loc[edf.index].values[mem_cols:].reshape(1,-1)).sum(axis=1)*mass_blowup).round(0).astype(np.int64)
    
    s=np.argsort(am)
    am,bm=am[s],bm[s]

            
    if flag_crats: minC,maxC=minC[s],maxC[s] 
    q=am<=bmax
    am,bm=am[q],bm[q]
    batches=len(am)
    print("array too large for memory, performing cartesian product in batches: "+str(batches))
    
    #re-calculate base rdbe and base dbe
    if flag_rdbe_max or flag_rdbe_min: 
        
        base_rdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
        if len(Xrdbe[Xrdbe<mem_cols]): base_rdbe +=(zm[:, Xrdbe].sum(axis=1)*2).astype(rdbe_bitlim)
        if len(Yrdbe[Yrdbe<mem_cols]): base_rdbe -=zm[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
        if len(Zrdbe[Zrdbe<mem_cols]): base_rdbe +=zm[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim)
    
        #batch rdbe
        batch_rdbeX,batch_rdbeY,batch_rdbeZ=Xrdbe-mem_cols,Yrdbe-mem_cols,Zrdbe-mem_cols
        batch_rdbeX,batch_rdbeY,batch_rdbeZ=batch_rdbeX[batch_rdbeX>-1],batch_rdbeY[batch_rdbeY>-1],batch_rdbeZ[batch_rdbeZ>-1]
        batch_rdbe=np.zeros(len(bm),dtype=rdbe_bitlim)
        if len( batch_rdbeX): batch_rdbe +=(bm[:, batch_rdbeX].sum(axis=1)*2).astype(rdbe_bitlim)
        if len( batch_rdbeY): batch_rdbe -=bm[:, batch_rdbeY].sum(axis=1).astype(rdbe_bitlim)
        if len( batch_rdbeZ): batch_rdbe +=bm[:, batch_rdbeZ].sum(axis=1).astype(rdbe_bitlim)
        
        
    if filt_LewisSenior: 
        base_dbe=np.sum(zm*vdf.loc[elements].values,axis=1)+2
        batch_dbe=np.sum(bm*vdf.loc[elements[mem_cols:]].values,axis=1)

    #precompute batch halogens
    if filt_halogens:    
        base_halo=(zm[:,halcols]>0).sum(axis=1)
        batch_halo=(bm[:,halcols[halcols>=mem_cols]-mem_cols]>0).sum(axis=1)
    
    #precompute batch multimetal
    if filt_multimetal:  
        base_metal=(zm[:,metcols]>0).sum(axis=1)
        batch_metal=(bm[:,metcols[metcols>=mem_cols]-mem_cols]>0).sum(axis=1)
        
    #calculate base mass frequencies
    mc=np.bincount(mass.astype(np.int64))    
    cmc=np.cumsum(mc)           
    um=np.argwhere(mc>0)[:,0]  #unique masses
    count_bit=bits(mc.max()*len(bm))

    #create memory mapping partitions
    partitions=int(np.ceil(len(bm)*sys.getsizeof(zm)/remRam)) #recalulate the nr of partitions
    memfiles=[unsorted_comp_output_path[:-4]+"_p"+str(i)+".npy" for i in range(partitions)]
    print(str(partitions)+ " Partitions ")

    #figure out correct partitions
    xs=np.linspace(0,len(mc)-1,1000).round(0).astype(int)
    vs=np.add.reduceat(mc,xs)
    czs=np.cumsum(vs*[np.sum(x>am) for x in xs])
    mass_ixs=np.hstack([np.interp(np.linspace(0,czs[-1],partitions+1),czs,xs).astype(int)])[1:-1]
    qtrim=len(zm)
    

    #%% Write unsorted array



    fsz=[]
    with ExitStack() as stack:
        files = [stack.enter_context(NpyAppendArray(fname, delete_if_exists=True) ) for fname in memfiles]
#%%

        for ib, b in enumerate(bm):
            
    
            b=bm[ib]
            
            print("writing unsorted batch: "+str(ib)+" ( "+str(np.round((ib+1)/batches*100,2))+" %)")
                
            #filter max mass
            q=(mass<=(bmax-am[ib]))
            if not q[-1]: qtrim=np.argmax(~q) 
            zm,mass=zm[:qtrim],mass[:qtrim] #truncate mass
            #if not len(mass): continue
                
            zm[:,mem_cols:]=bm[ib]
            

            ### find partitions
            umparts=[]
            if len(mass_ixs):
                x=mass_ixs-am[ib]-1
                q=x>0
                
                if not q.sum(): umparts=[0]*np.sum(~q) #end case
                else:
                    m=mass[cmc[x[q]]] #gives error
                    bs=np.clip(um[np.clip(np.vstack([find_closest(um,m)-1,find_closest(um,m)+1]).T,0,len(um)-1)],0,None)   
                    bs=find_closest(mass,bs)
                    ds=np.clip(np.diff(bs,axis=1).flatten(),0,len(zm)-1)
                    d=cm(zm[create_ranges(bs)])-np.repeat(mass_ixs[q],ds)>=0
                    ixs=np.hstack([0,np.cumsum(ds)])
                    umparts=np.hstack([[0]*np.sum(~q),bs[:,0]+np.array([np.argmax(d[ixs[i]:ixs[i+1]]) for i,_ in enumerate(ixs[:-1])])]).astype(int)
            umparts=np.hstack([0,umparts,len(zm)]).astype(int)
            
            ##### chemical filtering #####
            qr=np.ones(len(zm),bool)
     
            #C-centric ratio filtering
            if flag_crats: qr=qr&(zm[:,0]>=minC[ib]) &(zm[:,0]<=maxC[ib])
             
            #rdbe filtering
            if flag_rdbe_max or flag_rdbe_min: #or crdbe
                brdbe=base_rdbe[:qtrim]+batch_rdbe[ib]
                #global rdbe
                if flag_rdbe_min:               qr = qr & (brdbe >= (min_rdbe*2))
                if flag_rdbe_max:               qr = qr & (brdbe <= (max_rdbe*2))
                
                #C-centric rdbe
                if flag_crdbe:                  qr= qr & (brdbe>=Crdbes.values[zm[:,0]-Crdbes.index[0],0]) & (brdbe<=Crdbes.values[zm[:,0]-Crdbes.index[0],1])
                              
            #dbe filtering
            if filt_LewisSenior:                qr = qr & ((base_dbe[:qtrim]+batch_dbe[ib])%2==0)

            #chemical ratio filtering
            if len(batch_rats):
                for _,rat in batch_rats.iterrows():
                    r=zm[:,rat.lix]/zm[:,rat.rix]
                    qr=qr & ((~np.isfinite(r)) | ((r>=rat.low) & (r<=rat.high)))
                  
            # #NOPS filtering (slow)
            # if len(nops):
            #     for x,row in nops.iterrows():
            #         #qr=qr & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,row.ixs]<row.values[2:-1],axis=1)))  
            #         qr=qr & (((zm[:,row.ixs]==0).any(axis=1)) | (np.all(zm[:,row.ixs]<=row[row.els].values,axis=1)))
            #         qr=qr & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,row.ixs]<=row[row.els].values,axis=1)))
                    
            #7gr common element filtering
            if filt_7gr=="Common":
                xs=np.hstack([0,np.searchsorted(mass+am[ib],crep.index),len(qr)]) 
                qr=qr & np.hstack([np.all(zm[xs[ix]:xs[ix+1],mcx]<=row,axis=1) for ix,row in enumerate(crep.values)])
            
            #Multimetal filtering
            if filt_multimetal>0:
                if len(metcols):
                    qr=qr & (base_metal[:qtrim]+batch_metal[ib]<=filt_multimetal)              

            #Multihalogen filtering
            if filt_halogens>0:
                if len(halcols):
                    qr=qr & (base_halo[:qtrim]+batch_halo[ib]<=filt_halogens)  
                
            
            # #test
            # ls=[]
            # for p in range(partitions):
            #     l,r=umparts[p],umparts[p+1]
            #     ls.append(len(zm[l:r][qr[l:r]]))
            # plt.plot(ls)
            # ""+1
#%%
            ##### write in partitions #####
            for p in range(partitions):
                l,r=umparts[p],umparts[p+1]
                if r>l:
                    d=zm[l:r][qr[l:r]]
                    if len(d):
                        files[p].append(d)     
                
                #else: files[p].close() why close?
                

            
            
            
            #%%
            
    print("Completed")
    print(" ")
    


 
    #%% Write sorted table

    prev_mass,prev_mass_f,prev_comps=[],[],[]
    cur_ixs=0 
    borders=[] #test
    once=True #flag
    for p in np.arange(partitions):
        print("partition: "+str(p)+" ("+str(round(p/partitions*100,2))+" %)")
        
       
        if not os.path.exists(memfiles[p]):
            print("Warning: no compositions found for this partitions!")
            continue
            
        #%%    
        comps=np.load(memfiles[p])
        if not len(comps): continue
                            
        
        
        #precise float mass (memory efficient batched addition) 
        mi = np.zeros(len(comps), dtype=np.uint64)
        
        if write_mass: mf = np.zeros(len(comps), dtype=np.float32) #can be made full float64 for furter speedup
        remRam=maxRam-(mm.free-psutil.virtual_memory().free)
        stepsize=np.round(remRam/(a64*len(edf))/2,0).astype(int)
        ixs=np.arange(0,len(comps)+stepsize,stepsize)
        for im in range(len(ixs)-1):                
            f=np.sum(comps[ixs[im]:ixs[im+1]]*mdf.loc[edf.index].values.T,axis=1) #float calculation
            mi[ixs[im]:ixs[im+1]]=(f*mass_blowup).round(0).astype(int)           
            if write_mass: mf[ixs[im]:ixs[im+1]]=f
            del f
            
        # #NOPS filtering (slow, therefore moved to end)
        if len(nops):
            q  = np.ones(len(comps), dtype=bool) #mass overflow
            for x,row in nops.iterrows():
                q=q & ((~np.all(comps[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(comps[:,row.ixs]<=row[row.els].values,axis=1)))
            mi,comps=mi[q],comps[q]
            if write_mass: mf=mf[q]
  
        if not len(comps): continue
        uc=np.bincount(mi.astype(np.int64)).astype(count_bit)
        
        s=np.argsort(mi,kind="mergesort") #sort
        
        if partitions==1: #single partition
            
            #write
            with NpyAppendArray(comp_output_path, delete_if_exists=True) as fc:
                fc.append(comps[s])
            if write_mass: 
                with NpyAppendArray(mass_output_path, delete_if_exists=True) as fm:
                    fm.append(mf[s])

        else: #deal with roundoff error between partitions

            ss,se=int(uc[mi.min()]), int(uc[mi.max()]) #last mass index, first mass index
            
            if once: #not p: #first entry

                #write
                with NpyAppendArray(comp_output_path, delete_if_exists=True) as fc:
                    fc.append(comps[s[:-se]])
                if write_mass: 
                    with NpyAppendArray(mass_output_path, delete_if_exists=True) as fm:
                        fm.append(mf[s[:-se]])
                
                #store previous
                prev_comp,prev_mass=comps[s[-se:]],mi[s[-se:]]
                if write_mass: prev_fmass=mf[s[-se:]]
                
                once=False #turn off toggle
            
            
            else: #elif p: #second or higher partitions
                
        
                #presort with previous border
                cur_comp,cur_mass=comps[s[:ss]],mi[s[:ss]]
                ccomp,cmass=np.vstack([prev_comp,cur_comp]),np.hstack([prev_mass,cur_mass])
                sx=np.argsort(cmass)
                ccomp,cmass=ccomp[sx],cmass[sx]
                
                #add border
                if write_mass:
                    cur_fmass=mf[s[:ss]]
                    cfmass=np.hstack([prev_fmass,cur_fmass])[sx]
                    mf=np.hstack([cfmass,mf[s[ss:]]])  #this is now sorted comps 
                comps=np.vstack([ccomp,comps[s[ss:]]]) #this is now sorted mass

                if p<partitions-1:  
                                            
                    #store previous
                    prev_comp,prev_mass=comps[-se:],mi[-se:]
                    if write_mass: prev_fmass=mf[-se:]                   
                    
                    #write
                    with NpyAppendArray(comp_output_path, delete_if_exists=False) as fc:
                        fc.append(comps[:-se])
                    if write_mass: 
                        with NpyAppendArray(mass_output_path, delete_if_exists=False) as fm:
                            fm.append(mf[:-se])
                    
                    
                elif p==partitions-1: 
                    
                    #write
                    with NpyAppendArray(comp_output_path, delete_if_exists=False) as fc:
                        fc.append(comps)
                    if write_mass: 
                        with NpyAppendArray(mass_output_path, delete_if_exists=False) as fm:
                            fm.append(mf)
        
        
    
        if remove:
            del comps
            os.remove(memfiles[p])
            
        #flush, close, reopen
        emp[:len(uc),1]+=uc 
        emp.flush()
        emp._mmap.close()
        emp = open_memmap(m2g_output_path, mode="r+")
    
    #write final loop?
    
#%%
# efile="D:/231225_CartMFP/CartMFP_Database/C[58]H[108]N[13]P[3]O[25]S[5]F[28]Cl[7]Br[6]I[5]_b100000max800rdbe-5_32_7grCommon_m2g.npy"
# te=np.load(efile)

# file="D:/231225_CartMFP/CartMFP_Database/C[58]H[108]N[13]P[3]O[25]S[5]F[28]Cl[7]Br[6]I[5]_b100000max800rdbe-5_32_7grCommon_comp.npy"
#      #'D:/231225_CartMFP/CartMFP_Database/C[58]H[108]N[13]P[3]O[25]S[5]F[28]Cl[7]Br[6]I[5]_b100000max800rdbe-5_32_7grCommon_comp.npy'
# test=np.load(file,mmap_mode="r")

# testm=(test*edf.fmass.values).sum(axis=1)

# mfile="D:/231225_CartMFP/CartMFP_Database/C[58]H[108]N[13]P[3]O[25]S[5]F[28]Cl[7]Br[6]I[5]_b100000max800rdbe-5_32_7grCommon_mass.npy"
# mm=np.load(mfile,mmap_mode="r")



# #sorted

# c=np.bincount((testm*mass_blowup).round(0).astype(int))

# nz=np.argwhere(c)[:,0] #this is completely wrong?
# z=np.zeros(len(c),np.int64)
# z[nz]=np.cumsum(c[nz])-c[nz]

# teste=np.vstack([z,c]).T

# np.sum(np.diff(mm)<-0.1)                         #sorted mass
# np.sum(abs(mm-testm)>0.1)                        #mfile and calcm are the same
# (np.abs(te[:len(c)]-teste).sum(axis=1)>0).sum()  #also emp is equal to teste constructed

# #%%

# mv=((compounds.mass)*mass_blowup).round(0).astype(int).values
# mv=mv[mv<800*mass_blowup]
# mve=(mv.reshape(-1,1)+np.arange(-1,2))#.flatten()

# res=[]
# for ix,r in enumerate(mve):
#     x =te[r].astype(np.int64)
#     mr=np.arange(x[:,1].sum()) - np.repeat(np.cumsum(x[:,1])-x[:,1], x[:,1])
#     cq=np.repeat(x[:,0],x[:,1])+mr
    
    
    
#     cs=test[cq]
#     res.append(np.hstack([np.repeat(mv[ix]/mass_blowup,len(cs)).reshape(-1,1),cs]))
    
# res=np.vstack(res)

#when executed like this it works well?

#mve[np.in1d(mve,np.argwhere(c)[:,0])]
# test[np.all(test==np.array([10,13,5,0,1,0,0,0,0,0]),axis=1)] #row 60817 in comps, row in emp 21911201

#%% write index lookup table
        
if not need_batches:
    
    #### Filtering ####
    
    
    # #filter rdbe (already done)
    # if flag_rdbe_max or flag_rdbe_min or flag_crdbe:
        
    #     #base rdbe
    #     base_rdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
    #     if len(Xrdbe[Xrdbe<mem_cols]): base_rdbe +=(zm[:, Xrdbe].sum(axis=1)*2).astype(rdbe_bitlim)
    #     if len(Yrdbe[Yrdbe<mem_cols]): base_rdbe -=zm[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
    #     if len(Zrdbe[Zrdbe<mem_cols]): base_rdbe +=zm[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim)
    
    #     if flag_crdbe:
    #         q=(base_rdbe>=Crdbes.values[zm[:,0]-Crdbes.index[0],0]) & (base_rdbe<=Crdbes.values[zm[:,0]-Crdbes.index[0],1])
    #         zm,mass=zm[q],mass[q]
            

    #     #prefilter on base rdbe
    #     q=np.ones(len(mass),bool)
    #     if flag_rdbe_min: q=q & (base_rdbe>=min_rdbe)
    #     if flag_rdbe_max: q=q & (base_rdbe<=max_rdbe)
    #     mass,zm=mass[q],zm[q]
        
        
    # #filter chemical ratios [Golden rules #4,5] (already done)
    # if len(erats):
    #     q=np.ones(len(mass),bool)
    #     for _,rat in erats.iterrows():
    #         r=zm[:,rat.lix]/zm[:,rat.rix]
    #         q=q & ((~np.isfinite(r)) | ((r>=rat.low) & (r<=rat.high)))
    #     mass,zm=mass[q],zm[q]
        
  
    # #C-centric filtering (already done)
    # if flag_crats:
    #     q=(zm[:,1:mem_cols]>=countsl[edf.index[1:mem_cols]].values[zm[:,0]-countsl.index[0]]).all(axis=1)
    #     zm,mass=zm[q],mass[q]
        
    #     #filter maximum element counts
    #     q=(zm[:,1:mem_cols]<=countsh[edf.index[1:mem_cols]].values[zm[:,0]-countsh.index[0]]).all(axis=1)
    #     zm,mass=zm[q],mass[q]
    
    # #precompute NOPS probability [Golden rule #6] (already done)
    # if filt_NOPS:
    #     nops_ixs=np.hstack([np.argwhere(elements==e).flatten() for e in nops.columns[2:]])
    #     nops_vals=nops[nops.columns[2:]].values
        
    #     #prefilter on NOPS probabilities
    #     q=np.ones(len(mass),bool)
    #     for x,row in nops.iterrows():
    #         q=q & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,nops_ixs]<row.values[2:-1],axis=1)))  #OR above lim OR 
    #     mass,zm=mass[q],zm[q]
  

    #precompute dbe (LEWIS & SENIOR rules) [Golden rule #2]
    if filt_LewisSenior: #integer dbe
        base_dbe=np.sum(zm*vdf.loc[elements].values,axis=1)+2 #why +2 isnt that meaningless?
        q=(base_dbe%2)==0
        mass,zm=mass[q],zm[q]

    #7gr common element filtering
    if filt_7gr=="Common":
        xs=np.hstack([0,np.searchsorted(mass,crep.index),len(mass)]) 
        q = np.hstack([np.all(zm[xs[ix]:xs[ix+1],mcx]<row,axis=1) for ix,row in enumerate(crep.values)])
        mass,zm=mass[q],zm[q]

    #Multimetal filtering
    if filt_multimetal>0:
        if len(metcols):
            q=np.sum(zm[:,metcols]>0,axis=1)<=filt_multimetal  
            mass,zm=mass[q],zm[q]
    
    #Multihalogen filtering
    if filt_halogens>0:
        if len(halcols):
            q=np.sum(zm[:,halcols]>0,axis=1)<=filt_halogens     
            mass,zm=mass[q],zm[q]

    #### write outputs ####
    
    if write_mass: 
        mf = np.zeros(len(zm), dtype=np.float32) #can be made full float64 for furter speedup
        remRam=maxRam-(mm.free-psutil.virtual_memory().free)
        stepsize=np.round(remRam/(a64*len(edf))/2,0).astype(int)
        ixs=np.arange(0,len(zm)+stepsize,stepsize)
        for im in range(len(ixs)-1):                
            mf[ixs[im]:ixs[im+1]]=np.sum(zm[ixs[im]:ixs[im+1]]*mdf.loc[edf.index].values.T,axis=1) #float calculation   
        np.save(mass_output_path, mf)
    
    np.save(comp_output_path, zm)
    emp[:(mass.max()+1).astype(int),1]=np.bincount(mass.astype(np.int64))

nz=np.argwhere(emp[:,1])[:,0] #this is completely wrong?
emp[nz,0]=np.cumsum(emp[nz,1])-emp[nz,1]
emp.flush()


if sparse_index:

    import scipy.sparse as sp
    print("Making sparse matrix")
    n_rows, n_cols = emp.shape
    chunk_size = int(1e7)  # tune based on RAM
    
    sparse_blocks = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = emp[start:end]         
        sparse_chunk = sp.csr_matrix(chunk)
        sparse_blocks.append(sparse_chunk)
    sparse_matrix = sp.vstack(sparse_blocks, format="csr")
    sp.save_npz(m2g_output_path.replace(".npy",".npz"), sparse_matrix)

    del n_rows,n_cols,chunk
    emp._mmap.close()
    del emp
    if os.path.exists(m2g_output_path): 
        Path.unlink(m2g_output_path)
 


