# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:20:05 2024

@author: hkleikamp
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

#composition arguments
composition="H[200]C[75]N[50]O[50]P[10]S[10]"   # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]
max_mass = 1000         # default 1000
min_rdbe = -5           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
max_rdbe = 80

#advanced chemical rules
filt_7gr="Common"                                                                   #False,True,"Common","Extended",Toggles all advanced chemical filtering using rules #2,4,5,6 of Fiehn's "7 Golden Rules" 
filt_LewisSenior=True                                                               #Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence)
filt_ratios="HC[0,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]"    #Golden Rules #4,5: Filter on chemical ratios with extended range 99.9% coverage
filt_NOPS=True                                                                      #Golden Rules #6:   Filter on NOPS probabilities
filt_multimetal=1                                                                   #only allow a single metallic atom type
metals=["Na","Mg","Al","Si","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se",
        "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","Cs","Ba","la","Ce",
        "Pr","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb"]

#performance arguments
maxmem = 10e9 #0.7      # fraction of max free memory usage 
mass_blowup = 100000 #40000     # converting mass to int (higher blowup -> better precision, but higher memory usage)
write_mass=True                   # also writes float array of masses

#filepaths
mass_table = str(Path(basedir, "mass_table.tsv"))           # table containing element masses, default: CartMFP folder/ mass_table.tsv"
Cartesian_output_folder = str(Path(basedir, "Cart_Output")) # default: CartMFP folder / Cart_Output
Cartesian_output_file=""  

write_params=True #write an output file with arguments used to construct the db
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
    parser.add_argument("-folder_name", "--Cartesian_output_folder",  default=str(Path(basedir, "Cart_Output")),   required = False, help="Output folder for cartesian files")   
    parser.add_argument("-file_name",   "--Cartesian_output_file",   default="", required = False, help="Output file name for cartesian files")   
   
    #composition constraints
    parser.add_argument("-c", "--composition", default="H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]", 
    required = False, help="ALlowed elements and their minimum and maximum count. The following syntax is used: Element_name[minimum_count,maximum_count]")  
    parser.add_argument("-max_mass",  default=1000, required = False, help="maximum mass of compositions",type=float)  
    parser.add_argument("-min_rdbe",  default=-5,   required = False, help="minimum RDBE of compositions. set False to turn off",type=float)  
    parser.add_argument("-max_rdbe",  default=80,   required = False, help="maximum RBDE of compositions. set False to turn off",type=float)  

    #advanced composition constraints
    parser.add_argument("-filt_7gr",  default=True,   required = False, help="Toggles all advanced chemical filtering using rules #2,4,5,6 of Fiehn's 7 Golden Rules ")
    parser.add_argument("-filt_LewisSenior",  default=True,   required = False, help="Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence)")
    parser.add_argument("-filt_ratios",  default="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]" ,   required = False, help="Golden Rules #4,5: Filter on chemical ratios with extended range 99.9% coverage")  
    parser.add_argument("-filt_NOPS",  default=True,   required = False, help="Golden Rules #6:   Filter on NOPS probabilities")

    #performance arguments
    parser.add_argument("-mem",  default=0.7, required = False, help="if <=1: max fraction of available RAM used, if >1: mass RAM usage in GB",type=float)  
    parser.add_argument("-mass_blowup",  default=100000, required = False, help="multiplication factor to make masses integer. Larger values increase RAM usage but reduce round-off errors",type=int)  
    parser.add_argument("-write_mass",  default=True, required = False, help="Also create a lookup table for float masses",type=int)  
        
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
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


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



# %% Construct MFP space


# % Construct elemental space dataframe
edf=pd.DataFrame([i.replace(",","[").split("[") if "," in i else [i.split("[")[0],0,i.split("[")[-1]] for i in composition.split("]")[:-1]] ,columns=["symbol","low","high"]).set_index("symbol")

edf["low"]=pd.to_numeric(edf["low"],errors='coerce')
edf["high"]=pd.to_numeric(edf["high"],errors='coerce')
edf=edf.ffill(axis=1)


if filt_7gr=="Common": #max element count restriction (Wiley/DNP)
    crep=pd.DataFrame([[ 39, 72,20,20,9,10,16,10, 4,  8],
                       [ 78,126,25,27,9,14,34,12, 8, 14],
                       [156,236,32,63,9,14,48,12,10, 15],
                       [162,236,48,78,9,14,48,12,10, 15]],
                      index=[500,1000,2000,3000],
                      columns=["C","H","N","O","P","S","F","Cl","Br","Si"])
    
    for n,r in crep.iterrows():
        if n>=max_mass: 
            m=edf.merge(r,left_index=True,right_index=True,how="left")
            edf.loc[m.index,"high"]=m[n].values
            break
            
        

if edf.isnull().sum().sum(): #fill in missing values from composotion string.
    print("Warning! missing element maxima detected in composition. Imputing from maximum mass (this might affect performance)")
    edf.loc[edf["high"].isnull(),"high"]=(max_mass/mdf.loc[edf.index]).astype(int).values[edf["high"].isnull()].flatten()


edf[["low","high"]]=edf[["low","high"]].fillna(0).astype(int)
edf["high"]=np.clip(edf["high"],0,(max_mass/mdf.loc[edf.index]).astype(int)) #limit to max mass


edf = edf.sort_values(by="high", ascending=False)
edf["arr"] = edf.apply(lambda x: np.arange(
    x.loc["low"], x.loc["high"]+1), axis=1)
edf["fmass"] = mdf.loc[edf.index] #float mass
edf["mass"]  = (edf["fmass"]*mass_blowup).astype(np.uint64)
elements=edf.index.values
params["elements"]=elements.tolist()
metcols=np.hstack([np.argwhere(elements==m)[:,0] for m in metals])

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

if filt_7gr=="Common":
    filt_LewisSenior=True
    filt_nops=True
    filt_ratios="HC[0.2,3.1]FC[0,1.5]ClC[0,0.8]BrC[0,0.8]NC[0,1.3]OC[0,1.2]PC[0,0.3]SC[0,0.8]SiC[0,0.5]" #common ratio range
    min_rdbe=0
    
if filt_7gr=="Extended":
    filt_LewisSenior=True
    filt_nops=True
    filt_ratios="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]" #extended ratio range

if not filt_7gr: #turn of 7gr [except for custom filt_ratios]
    filt_LewisSenior=False
    filt_nops=False
    if filt_ratios=="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]": filt_ratios=False
    
    



if not "[" in filt_ratios: filt_ratios=False
    
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


#%% construct output paths

if Cartesian_output_file=="":
    Cartesian_output_file = "".join(edf.index+"["+edf.low.astype(str)+","+edf.high.astype(
        str)+"]")+"_b"+str(mass_blowup)+"max"+str(int(max_mass))+"rdbe"+str(min_rdbe)+"_"+str(max_rdbe) 
    if filt_7gr: Cartesian_output_file+="_7gr"
    if filt_7gr=="Common":      Cartesian_output_file+="Common"
    elif filt_7gr=="Extended":  Cartesian_output_file+="Extended"
    elif filt_ratios!="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]": Cartesian_output_file+="_customfilt"
    Cartesian_output_file=Cartesian_output_file.replace("[0,","[")
else: 
    write_params=True


if not len(Cartesian_output_folder):
    Cartesian_output_folder = os.getcwd()
else:
    if not os.path.exists(Cartesian_output_folder):
        os.makedirs(Cartesian_output_folder)

basepath=str( Path(Cartesian_output_folder, Cartesian_output_file))
unsorted_comp_output_path =basepath+"_unsorted_comp.npy"
mass_output_path =         basepath+"_mass.npy"
comp_output_path =         basepath+"_comp.npy"
m2g_output_path  =         basepath+"_m2g.npy"

if write_params: 
    import json
    params.update({"column_masses":edf.fmass.values}) #add column masses
    with open(basepath+".params", 'w') as f:
        json.dump(params, f)

print("Output Cartesian file:")
print(Cartesian_output_file)
print("")


# Chemical filtering
flag_rdbe_min = type(min_rdbe) == float or type(min_rdbe) == int
flag_rdbe_max = type(max_rdbe) == float or type(max_rdbe) == int


if flag_rdbe_min or flag_rdbe_max:
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

if zm.max()<256: bitlim=np.uint8
    
# add room for remaining columns
zm = np.hstack(
    [zm, np.zeros([len(zm), len(edf)-mem_cols], dtype=bitlim)])
s = np.argsort(mass,kind="mergesort")  
mass, zm = mass[s], zm[s]


# compute cartesian batches
print("")
if need_batches:

    batches = reduce(operator.mul, edf.iloc[mem_cols:]["high"].values+1, 1)
    
    # compute cartesian product of the remaining elements
    arrays = edf.arr.values[mem_cols:].tolist()
    print("")
    print("computing remaining cartesian:")
    bm = cartesian(arrays,bitlim=bitlim)
    am = ((bm*mdf.loc[edf.index].values[mem_cols:].reshape(1,-1)).sum(axis=1)*mass_blowup).round(0).astype(np.int64) 
    s=np.argsort(am)
    am,bm=am[s],bm[s]
    q=am<=bmax
    am,bm=am[q],bm[q]
    batches=len(am)
    print("array too large for memory, performing cartesian product in batches: "+str(batches))
    
print("")

#%% Write unsorted array

emp = open_memmap(m2g_output_path, mode="w+", shape=(bmax+1*mass_blowup,2),dtype=bits(bmax))

if need_batches:

    
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
        if flag_rdbe_max: q=q & ((base_rdbe-batch_rdbe.min())<=max_rdbe)
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
            
    #precompute NOPS probability [Golden rule #6]
    if len(nops):
        nops_ixs=np.hstack([np.argwhere(elements==e).flatten() for e in nops.columns[2:]])
        nops_vals=nops[nops.columns[2:]].values
        
        #prefilter on NOPS probabilities
        q=np.ones(len(mass),bool)
        for x,row in nops.iterrows():
            q=q & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,nops_ixs]<row.values[2:-1],axis=1)))  #OR above lim OR 
            
        mass,zm=mass[q],zm[q]

    #re-calculate base rdbe and base dbe
    if flag_rdbe_max or flag_rdbe_min:
        base_rdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
        if len(Xrdbe[Xrdbe<mem_cols]): base_rdbe +=(zm[:, Xrdbe].sum(axis=1)*2).astype(rdbe_bitlim)
        if len(Yrdbe[Yrdbe<mem_cols]): base_rdbe -=zm[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
        if len(Zrdbe[Zrdbe<mem_cols]): base_rdbe +=zm[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim)
        
    if filt_LewisSenior: base_dbe=np.sum(zm*vdf.loc[elements].values,axis=1)+2

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

    #%%
    with ExitStack() as stack:
        files = [stack.enter_context(NpyAppendArray(fname, delete_if_exists=True) ) for fname in memfiles]


        for ib, b in enumerate(bm):
            print("writing unsorted batch: "+str(ib)+" ( "+str(np.round((ib+1)/batches*100,2))+" %)")
                
            #filter max mass
            q=(mass<=(bmax-am[ib]))
            if not q[-1]: qtrim=np.argmax(~q) 
            zm,mass=zm[:qtrim],mass[:qtrim] #truncate mass
        
            zm[:,mem_cols:]=bm[ib]
            
            ### find partitions
            umparts=[]
            if len(mass_ixs):
                x=mass_ixs-am[ib]-1
                q=x>0
                
                if not q.sum(): umparts=[0]*np.sum(~q) #end case
                else:
                    m=mass[cmc[x[q]]]
                    bs=np.clip(um[np.clip(np.vstack([find_closest(um,m)-1,find_closest(um,m)+1]).T,0,len(um)-1)],0,None)   
                    bs=find_closest(mass,bs)
                    ds=np.clip(np.diff(bs,axis=1).flatten(),0,len(zm)-1)
                    d=cm(zm[create_ranges(bs)])-np.repeat(mass_ixs[q],ds)>=0
                    ixs=np.hstack([0,np.cumsum(ds)])
                    umparts=np.hstack([[0]*np.sum(~q),bs[:,0]+np.array([np.argmax(d[ixs[i]:ixs[i+1]]) for i,_ in enumerate(ixs[:-1])])]).astype(int)
            umparts=np.hstack([0,umparts,len(zm)]).astype(int)
            
      
            ##### chemical filtering #####
        
            qr=np.ones(len(zm),bool)
            
            #rdbe filtering
            if flag_rdbe_max or flag_rdbe_min:
                brdbe=base_rdbe[:qtrim]+batch_rdbe[ib]
                if flag_rdbe_min:               qr = qr & (brdbe >= (min_rdbe*2))
                if flag_rdbe_max:               qr = qr & (brdbe <= (max_rdbe*2))    
            
            #dbe filtering
            if filt_LewisSenior:                qr = qr & ((base_dbe[:qtrim]+batch_dbe[ib])%2==0)

            #chemical ratio filtering
            if len(batch_rats):
                for _,rat in batch_rats.iterrows():
                    r=zm[:,rat.lix]/zm[:,rat.rix]
                    qr=qr & ((~np.isfinite(r)) | ((r>=rat.low) & (r<=rat.high)))
                  
            #NOPS filtering
            if len(nops):
                for x,row in nops.iterrows():
                    qr=qr & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,nops_ixs]<row.values[2:-1],axis=1)))   
                    
            #7gr common element filtering
            if filt_7gr=="Common":
                xs=np.hstack([0,np.searchsorted(mass+am[ib],crep.index),len(qr)]) 
                qr=qr & np.hstack([np.all(zm[xs[ix]:xs[ix+1],mcx]<row,axis=1) for ix,row in enumerate(crep.values)])
                      
            #Multimetal filtering
            if filt_multimetal>0:
                if len(metcols):
                    qr=qr & np.sum(zm[:,metcols]>0,axis=1)<=filt_multimetal              

            ##### write in partitions #####
            for p in range(partitions):
                l,r=umparts[p],umparts[p+1]
                if r>l:
                    files[p].append(zm[l:r][qr[l:r]])     
                else: files[p].close()
                
            
    print("Completed")
    print(" ")
    


 
    #%% Write sorted table

    prev_mass,prev_mass_f,prev_comps=[],[],[]
    cur_ixs=0 
    borders=[]
   
    with ExitStack() as stack:
        
        fc=stack.enter_context(NpyAppendArray(comp_output_path, delete_if_exists=True))
        if write_mass: fm=stack.enter_context(NpyAppendArray(mass_output_path, delete_if_exists=True))
        
    
        for p in np.arange(partitions):
            print("partition: "+str(p)+" ("+str(round(p/partitions*100,2))+" %)")
            
            if not os.path.exists(memfiles[p]):
                print("Warning: no compositions found for this partitions!")
                continue
                
                
            comps=np.load(memfiles[p])
            
            
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
        
            uc=np.bincount(mi.astype(np.int64)).astype(count_bit)
            emp[:len(uc),1]+=uc
            s=np.argsort(mi,kind="mergesort") #sort

            if partitions==1:
                fc.append(comps[s])
                if write_mass: fm.append(mf[s])
            else: #deal with roundoff error between partitions


                if p: #sort combine -> write
                    cur_ixs=uc[mi[s[0]]]
                    cur_mass,cur_comps=mi[s[:cur_ixs]],comps[s[:cur_ixs]]
            
                    cmass,ccomps=np.hstack([cur_mass,prev_mass]),np.vstack([cur_comps,prev_comps])
                    ss=np.argsort(cmass)
                    fc.append(ccomps[ss])
                    
                    if write_mass:
                        cur_mass_f=mf[s[:cur_ixs]]
                        cmass_f=np.hstack([cur_mass_f,prev_mass_f])
                        fm.append(cmass_f[ss]) 
                        
                if p<partitions-1:  
                    prev_ixs=uc[-1]
                    prev_mass,prev_comps=mi[s[-1*prev_ixs:]],comps[s[-1*prev_ixs:]]
                    fc.append(comps[s[cur_ixs:-1*prev_ixs]])
                    
                    if write_mass: 
                        prev_mass_f=mf[s[-1*prev_ixs:]]
                        fm.append(mf[s[cur_ixs:-1*prev_ixs]])
                
                if p==partitions-1: 
                    fc.append(comps[s[cur_ixs:]])
                    if write_mass: fm.append(mf[s[cur_ixs:]])

            if remove:
                del comps
                os.remove(memfiles[p])
                
            #flush, close, reopen
            emp.flush()
            emp._mmap.close()
            emp = open_memmap(m2g_output_path, mode="r+")
            
        fc.close()
        fm.close()
    



#%% write index lookup table
        
if not need_batches:
    
    #### Filtering ####
    
    
    #precompute rdbe
    if flag_rdbe_max or flag_rdbe_min:
        
        #base rdbe
        base_rdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
        if len(Xrdbe[Xrdbe<mem_cols]): base_rdbe +=(zm[:, Xrdbe].sum(axis=1)*2).astype(rdbe_bitlim)
        if len(Yrdbe[Yrdbe<mem_cols]): base_rdbe -=zm[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
        if len(Zrdbe[Zrdbe<mem_cols]): base_rdbe +=zm[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim)

        #prefilter on base rdbe
        q=np.ones(len(mass),bool)
        if flag_rdbe_min: q=q & (base_rdbe>=min_rdbe)
        if flag_rdbe_max: q=q & (base_rdbe<=max_rdbe)
        mass,zm=mass[q],zm[q]
        
  
    #precompute dbe (LEWIS & SENIOR rules) [Golden rule #2]
    if filt_LewisSenior: #integer dbe
        base_dbe=np.sum(zm*vdf.loc[elements].values,axis=1)+2 #why +2 isnt that meaningless?
        q=(base_dbe%2)==0
        mass,zm=mass[q],zm[q]
    
    #precompute chemical ratios [Golden rules #4,5]
    if len(erats):
        q=np.ones(len(mass),bool)
        for _,rat in erats.iterrows():
            r=zm[:,rat.lix]/zm[:,rat.rix]
            q=q & ((~np.isfinite(r)) | ((r>=rat.low) & (r<=rat.high)))
        mass,zm=mass[q],zm[q]
            
    #precompute NOPS probability [Golden rule #6]
    if filt_NOPS:
        nops_ixs=np.hstack([np.argwhere(elements==e).flatten() for e in nops.columns[2:]])
        nops_vals=nops[nops.columns[2:]].values
        
        #prefilter on NOPS probabilities
        q=np.ones(len(mass),bool)
        for x,row in nops.iterrows():
            q=q & ((~np.all(zm[:,nops.loc[x,"ixs"]]>row.lim,axis=1)) | (np.all(zm[:,nops_ixs]<row.values[2:-1],axis=1)))  #OR above lim OR 
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

    #### write outputs ####
    np.save(comp_output_path, zm)
    emp[:(mass.max()+1).astype(int),1]=np.bincount(mass.astype(np.int64))


nz=np.argwhere(emp[:,1])[:,0] #this is completely wrong?
emp[nz,0]=np.cumsum(emp[nz,1])-emp[nz,1]
emp.flush()

#%% test


# m=np.load(mass_output_path)
# comps=np.load(comp_output_path)
# abs(m-np.sum(comps*mdf.loc[elements].values,axis=1)).max()

# comps = np.load(comp_output_path, mmap_mode="r")
# m = np.load(mass_output_path, mmap_mode="r") #test
# abs(m-np.sum(comps*mdf.loc[elements].values,axis=1)).max()
