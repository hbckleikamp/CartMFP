# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:20:05 2024

@author: hkleikamp
"""



#%% Modules

from inspect import getsourcefile
import time
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

# %% change directory to script directory (should work on windows and mac)

basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())

# %% Arguments

#composition arguments
composition="H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]"   # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]
max_mass = 1000         # default 1000
min_rdbe = -5           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
max_rdbe = 80


#performance arguments
maxmem = 0.7            # fraction of max free memory usage 
mass_blowup = 40000     # converting mass to int (higher blowup -> better precision, but higher memory usage)
keep_all    = False     # also display mass/ adduct combinations for which no molecular formula was found

#filpaths
mass_table = str(Path(basedir, "mass_table.tsv"))           # table containing element masses, default: CartMFP folder/ mass_table.tsv"
Cartesian_output_folder = str(Path(basedir, "Cart_Output")) # default: CartMFP folder / Cart_Output


remove=True #removes unsorted composition file and sorted index file after building sorted array
debug=False #True
#%% Arguments for execution from command line.

if not hasattr(sys,'ps1'): #checks if code is executed from command line
    
    import argparse

    parser = argparse.ArgumentParser(
                        prog='CartMFP-space2cart',
                        description='molecular formula prediction, see: https://github.com/hbckleikamp/CartMFP')
    
    #output and utility filepaths
    parser.add_argument("-mass_table",                 default=str(Path(basedir, "mass_table.tsv")), required = False, help="list of element masses")  
    parser.add_argument("-cart_out", "--Cart_Output",  default=str(Path(basedir, "Cart_Output")), required = False, help="Output folder for cartesian files")   

    #composition constraints
    parser.add_argument("-c", "--composition", default="H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]", 
    required = False, help="ALlowed elements and their minimum and maximum count. The following syntax is used: Element_name[minimum_count,maximum_count]")  
    parser.add_argument("-max_mass",  default=1000, required = False, help="maximum mass of compositions")  
    parser.add_argument("-min_rdbe",  default=-5,   required = False, help="minimum RDBE of compositions. set False to turn off")  
    parser.add_argument("-min_rdbe",  default=80,   required = False, help="maximum RBDE of compositions. set False to turn off")  


    #performance arguments
    parser.add_argument("-mem",  default=0.7, required = False, help="if <=1: max fraction of available RAM used, if >1: mass RAM usage in GB")  
    parser.add_argument("-mass_blowup",  default=40000, required = False, help="multiplication factor to make masses integer. Larger values increase RAM usage but reduce round-off errors")  
    
    
    parser.add_argument("-remove",  default=True, required = False, help="removes unsorted composition file and sorted index file after building sorted array")  
    parser.add_argument("-d","--debug",  default=False, required = False, help="")  
    

    args = parser.parse_args()
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    
    print("")
    print(args) 
    print("")
    locals().update(args)



# %% General functions

# Source: Eli Korvigo (https://stackoverflow.com/questions/28684492/numpy-equivalent-of-itertools-product/28684982)
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


#read table with dynamic delmiter detection
def read_table(tabfile, *,
               Keyword="mass",
               ):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        try:
            tab = pd.read_excel(tabfile, engine='openpyxl')
        except:
            with open(tabfile, "r") as f:
                tab = pd.DataFrame(f.read().splitlines())

        # dynamic delimiter detection: if file delimiter is different, split using different delimiters until the desired column name is found
        if Keyword:
            if Keyword not in tab.columns:
                delims = [i[0] for i in Counter(
                    [i for i in str(tab.iloc[0]) if not i.isalnum()]).most_common()]
                delims=[i for i in delims if i!='"']
                for delim in delims:
                    if delim == " ":
                        delim = "\s"
                    try:
                        tab = pd.read_csv(tabfile, sep=delim)
                        if Keyword in tab.columns:
                            return tab
                    except:
                        pass

    return tab


#The number of bits needed to represent an integer n is given by rounding down log2(n) and then adding 1
def bits(x,neg=False):
    bitsize=np.array([8,16,32,64])
    
    dtypes=[np.uint8,np.uint16,np.uint32,np.uint64]
    if neg: dtypes=[np.int8,np.int16,np.int32,np.int64]
    
    return dtypes[np.argwhere(bitsize-(np.log2(x)+1)>0).min()]



# %% Get elemental metadata (or replace with utils table)


if os.path.exists(mass_table):
    mdf = pd.read_csv(mass_table, index_col=[0], sep="\t")

else:  # retrieve up to date mass table from nist
    print("no element mass table found, attempting to retrieve NIST data online.")
    print("")
    url = "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl"
    tables = pd.read_html(url)[0]

    # remove fully nan rows and columns
    tables = tables[~tables.isnull().all(axis=1)]
    tables = tables[tables.columns[~tables.isnull().all(axis=0)]]

    tables.columns = ["atomic_number", "symbol", "mass_number", 'Relative Atomic Mass',
                      'Isotopic  Composition', 'Standard Atomic Weight', 'Notes']
    tables = tables[["atomic_number", "symbol", "mass_number", 'Relative Atomic Mass',
                     'Isotopic  Composition', 'Standard Atomic Weight']]

    # remove deuterium and tritium trivial names
    tables.loc[tables["atomic_number"] == 1, "symbol"] = "H"
    tables = tables[tables['Isotopic  Composition'].notnull()
                    ].reset_index(drop=True)

    # floatify mass and composition
    for i in ['Relative Atomic Mass',
              'Isotopic  Composition', 'Standard Atomic Weight']:
        tables[i] = tables[i].str.replace(" ", "").str.replace(
            "(", "").str.replace(")", "").str.replace(u"\xa0", u"")
    tables[['Relative Atomic Mass', 'Isotopic  Composition']] = tables[[
        'Relative Atomic Mass', 'Isotopic  Composition']].astype(float)
    tables['Standard Atomic Weight'] = tables['Standard Atomic Weight'].str.strip(
        "[]").str.split(",")

    mdf = tables.sort_values(by=["symbol", 'Isotopic  Composition'], ascending=False).groupby(
        "symbol", sort=False).nth(0)[['symbol', 'Relative Atomic Mass']]
    mdf = mdf.set_index("symbol")
    mdf.columns = ["mass"]

    mdf.to_csv("mass_table.tsv", sep="\t")


# %% Construct MFP space

start_time = time.time()

# % Construct elemental space dataframe
edf=pd.DataFrame([i.replace("[",",").split(',') for i in  composition.split("]")[:-1]],columns=["symbol","low","high"]).set_index("symbol")

edf["low"]=pd.to_numeric(edf["low"],errors='coerce')
edf["high"]=pd.to_numeric(edf["high"],errors='coerce')
edf=edf.ffill(axis=1)

if edf.isnull().sum().sum(): #fill in missing values from composotion string.
    print("Warning! missing element maxima detected in composition. Imputing from maximum mass (this might affect performance)")
    edf.loc[edf["high"].isnull(),"high"]=(max_mass/mdf.loc[edf.index]).astype(int).values[edf["high"].isnull()].flatten()


edf[["low","high"]]=edf[["low","high"]].fillna(0).astype(int)
edf = edf.sort_values(by="high", ascending=False)
edf["arr"] = edf.apply(lambda x: np.arange(
    x.loc["low"], x.loc["high"]+1), axis=1)
edf["mass"] = (mdf.loc[edf.index]*mass_blowup).astype(np.uint64)
elements=edf.index.values


bitlim=np.uint8
if edf.high.max()>255: 
    bitlim=np.uint16
    print("element cound above 255 detected, using 16bit compositions")
    print("")


# % Determine number of element batches
mm = psutil.virtual_memory()
dpoints = np.array([10, 100, 1e3, 1e4, 1e5, 1e6]).astype(int)

# size of uint8 array
onesm = np.array([sys.getsizeof(np.ones(i).astype(bitlim)) for i in dpoints])
a8, b8 = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

# size of uint64 array
onesm = np.array([sys.getsizeof(np.ones(i).astype(np.uint64))
                 for i in dpoints])
a64, b64 = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

rows = 1
memories = []
cols = len(edf)
for i in edf.high:
    rows *= i
    s_uint8 = rows*cols*a8[0]+b8[0]
    s_uint64 = rows*a64[0]+b64[0] 
    memories.append(s_uint8+s_uint64)
 

mem_cols = (np.argwhere(np.array(memories) < (mm.free*maxmem))[-1]+1)[0]
need_batches = len(edf)-mem_cols


# construct output path
Cartesian_output_file = "".join(edf.index+"["+edf.low.astype(str)+","+edf.high.astype(
    str)+"]")+"_b"+str(mass_blowup)+"max_"+str(int(max_mass))
Cartesian_output_file=Cartesian_output_file.replace("[0,","[")
if not len(Cartesian_output_folder):
    Cartesian_output_folder = os.getcwd()
else:
    if not os.path.exists(Cartesian_output_folder):
        os.makedirs(Cartesian_output_folder)

unsorted_comp_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_unsorted_comp.npy"

sort_index_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_sort_index.npy"

comp_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_comp.npy"

m2g_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_m2g.npy"




print("Output Cartesian file:")
print(Cartesian_output_file)
print("")


# Chemical filtering
flag_rdbe_min = type(min_rdbe) == float or type(min_rdbe) == int
flag_rdbe_max = type(max_rdbe) == float or type(max_rdbe) == int

# #chemical filtering for RDBE (it is int rounded so pick a generous range)
# #RDBE = X -Y/2+Z/2+1 where X=C Y=H or halogen, Z=N or P https://fiehnlab.ucdavis.edu/projects/seven-golden-rules/ring-double-bonds
Xrdbe = np.argwhere(edf.index == "C").flatten()
Yrdbe = np.argwhere(edf.index.isin(["H", "F", "Cl", "Br", "I"]))[0].flatten()
Zrdbe = np.argwhere(edf.index.isin(["N", "P"])).flatten()

bmax = int(np.round(max_mass*mass_blowup, 0))
#bmin = int(np.round(min_mass*mass_blowup, 0))

# % Compute base cartesian
print("constructing base cartesian:")
arrays = edf.arr.values[:mem_cols].tolist()
zm = cartesian(arrays)

mass = np.zeros(len(zm), dtype=np.uint64)
for i in range(zm.shape[1]):
    mass += edf.mass.values[i]*zm[:, i].astype(np.uint64)

# filter base cartesian on maximum mass
if max_mass:  # (recommended!)
    zm = zm[mass <= max_mass*mass_blowup]
    mass = mass[mass <= max_mass*mass_blowup]

if zm.max()<256: bitlim=np.uint8
    
# add room for remaining columns
zm = np.hstack(
    [zm, np.zeros([len(zm), len(edf)-mem_cols], dtype=bitlim)])
s = np.argsort(mass)
mass, zm = mass[s], zm[s]

# compute cartesian batches
print("")
if need_batches:

    batches = reduce(operator.mul, edf.iloc[mem_cols:]["high"].values+1, 1)
    print("array too large for memory, performing cartesian product in batches: "+str(batches))

    # compute cartesian product of the remaining elements
    arrays = edf.arr.values[mem_cols:].tolist()
    print("")
    print("computing remaining cartesian:")
    bm = cartesian(arrays)
    am = (bm*edf[mem_cols:].mass.values).sum(axis=1)  #filter bm based on max mass
    q=am<=bmax
    
print("")

#%% Write unsorted array



um,uc=np.unique(mass,return_counts=True) #can be done with diff count
zs=np.zeros(bmax+1,dtype=bits(uc.max()*len(bm)))



#%%

bls=[] #batch_lengths
with NpyAppendArray(unsorted_comp_output_path, delete_if_exists=True) as fc, NpyAppendArray(sort_index_output_path, delete_if_exists=True) as fs:

    for ib, b in enumerate(bm):
        print("writing unsorted batch: "+str(ib)+" ( "+str(np.round((ib+1)/batches*100,2))+" %)")
        

        zm[:,mem_cols:]=b
        m=mass+am[ib]
        
        
        #improve speed here!
        #max mass filtering
        q=np.ones(len(zm),dtype=bool)
        if max_mass: q=q & (m<=bmax)
    
        #RDBE filtering
        rdbe=np.ones(len(zm),dtype=np.int8)
        if len(Xrdbe): rdbe+=zm[:,Xrdbe][:,0]
        if len(Yrdbe): rdbe-=(zm[:,Yrdbe][:,0]/2).astype(int) #not fully accurate
        if len(Zrdbe): rdbe+=(zm[:,Zrdbe][:,0]/2).astype(int) #not fully accurate
                    
        if   flag_rdbe_min & flag_rdbe_max: q=q & (rdbe>=min_rdbe) &( rdbe<=max_rdbe)
        elif flag_rdbe_min:                 q=q & (rdbe>=min_rdbe)
        elif flag_rdbe_max:                 q=q & (rdbe<=max_rdbe)
    
        mq=m[q]
        um,uc=np.unique(mq,return_counts=True) #can be done with diff count (faster?)
        zs[um]+=uc.astype(np.uint32) 
        batch_len=q.sum()
        fs.append(np.zeros(batch_len,dtype=bits(len(mass))))  #append to placeholder sort index 
        fc.append(zm[q])                                      #append to unsorted mass array 
        bls.append(batch_len)


print("")



#%% Create sorting index



sort_ixs  = np.load(sort_index_output_path, mmap_mode="r+")
comps=np.load(unsorted_comp_output_path,mmap_mode="r")

dt=bits(len(comps)) #bits(np.sum(zs),neg=True)
czs=np.cumsum(zs,dtype=dt) 

sort_batches=np.hstack([0,np.cumsum(bls),len(comps)]).astype(dt)



for i in range(len(sort_batches)-1):
    print("writing sorted index: "+str(i)+" ( "+str(np.round((sort_batches[i+1])/len(comps)*100,2))+" %)")
    
    bcomps=comps[sort_batches[i]:sort_batches[i+1]]
    m=np.sum(bcomps*edf.mass.values,axis=1)
    
    um,uc=np.unique(m,return_counts=True)
    czs[um]-=uc.astype(dt)
     
    df=pd.DataFrame(np.diff(m)==0,columns=['bool'])
    cs= df['bool'].cumsum()
    c_arr=np.hstack([0,cs.sub(cs.where(~df['bool']).ffill().fillna(0)).values]).astype(dt)
    
    sort_ixs[(czs[m]+c_arr)]=np.arange(sort_batches[i],sort_batches[i+1],dtype=dt) #np.arange(len(m))+inc


print("")


    


#%% Write to sorted mass array


dzs=np.zeros(bmax+1,dtype=bits(zs.max()*len(bm)))
with NpyAppendArray(comp_output_path, delete_if_exists=True) as fc:

    for i in range(len(sort_batches)-1):
        
        print("writing sorted compositions: "+str(i)+" ( "+str(np.round((sort_batches[i+1])/len(comps)*100,2))+" %)")
        
        bcomps=comps[sort_ixs[sort_batches[i]:sort_batches[i+1]]]
        m=np.sum(bcomps*edf.mass.values,axis=1)

        fc.append(bcomps)
        um,uc=np.unique(m,return_counts=True)
        dzs[um]+=uc.astype(np.uint32)  #update datatype!! ????

        
#%% Cleanup

del comps,sort_ixs 

if remove:
    os.remove(unsorted_comp_output_path)
    os.remove(sort_index_output_path)



#%% Construct mass index

cdzs=np.cumsum(dzs)
emp=np.vstack([cdzs-dzs,cdzs,dzs]).T
np.save(m2g_output_path,emp)



