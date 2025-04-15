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
from numpy.lib.format import open_memmap
import tracemalloc
from contextlib import ExitStack
import matplotlib.pyplot as plt

# %% change directory to script directory (should work on windows and mac)

basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())

# %% Arguments

#composition arguments
#800
composition="H[0,160]C[0,60]N[0,40]O[0,40]P[0,8]S[0,8]"
max_mass=800
max_rdbe=68.0
min_rdbe=-5

# composition="H[200]C[75]N[50]O[50]P[10]S[10]"   # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]
# max_mass = 1000         # default 1000
# min_rdbe = -5           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
# max_rdbe = 80

# #1200
# composition="H[0,240]C[0,90]N[0,60]O[0,60]P[0,12]S[0,12]" 
# max_rdbe=92.0
# max_mass=1200


# # # #1400
# composition="H[0,280]C[0,105]N[0,70]O[0,70]P[0,14]S[0,14]"
# max_rdbe=104.0
# max_mass=1400

# # #1600
# composition="H[0,320]C[0,120]N[0,80]O[0,80]P[0,16]S[0,16]" 
# max_rdbe=116.0 
# max_mass=1600

# # #1800
# composition= "H[0,360]C[0,135]N[0,90]O[0,90]P[0,18]S[0,18]" 
# max_rdbe=128.0
# max_mass=1800

#2000
composition= "H[0,400]C[0,150]N[0,100]O[0,100]P[0,20]S[0,20]"
max_rdbe=140.0
max_mass=2000




#performance arguments
maxmem = 10e9 #0.7      # fraction of max free memory usage 
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
    parser.add_argument("-max_mass",  default=1000, required = False, help="maximum mass of compositions",type=float)  
    parser.add_argument("-min_rdbe",  default=-5,   required = False, help="minimum RDBE of compositions. set False to turn off",type=float)  
    parser.add_argument("-max_rdbe",  default=80,   required = False, help="maximum RBDE of compositions. set False to turn off",type=float)  


    #performance arguments
    parser.add_argument("-mem",  default=0.7, required = False, help="if <=1: max fraction of available RAM used, if >1: mass RAM usage in GB",type=float)  
    parser.add_argument("-mass_blowup",  default=40000, required = False, help="multiplication factor to make masses integer. Larger values increase RAM usage but reduce round-off errors",type=int)  
    
    
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

#https://stackoverflow.com/questions/41833740/numpy-index-of-the-maximum-with-reduction-numpy-argmax-reduceat
def numpy_argmin_reduceat(a,b):
    n = a.max()+1  # limit-offset
    grp_count = np.append(b[1:] - b[:-1], a.size - b[-1])
    shift = n*np.repeat(np.arange(grp_count.size), grp_count)
    return (a+shift).argsort()[b]

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

tracemalloc.start() 
cartesian_time = time.time()

# % Construct elemental space dataframe
edf=pd.DataFrame([i.replace(",","[").split("[") if "," in i else [i.split("[")[0],0,i.split("[")[-1]] for i in composition.split("]")[:-1]] ,columns=["symbol","low","high"]).set_index("symbol")

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

#%%

# construct output path
Cartesian_output_file = "".join(edf.index+"["+edf.low.astype(str)+","+edf.high.astype(
    str)+"]")+"_b"+str(mass_blowup)+"max"+str(int(max_mass))+"rdbe"+str(min_rdbe)+"_"+str(max_rdbe) 

Cartesian_output_file=Cartesian_output_file.replace("[0,","[")
if not len(Cartesian_output_folder):
    Cartesian_output_folder = os.getcwd()
else:
    if not os.path.exists(Cartesian_output_folder):
        os.makedirs(Cartesian_output_folder)

unsorted_comp_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_unsorted_comp.npy"

unsorted_mass_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_unsorted_mass.npy"

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

#%% batched addition
mass = np.zeros(len(zm), dtype=np.uint64)


#memory efficient batched addition 
remRam=maxRam-(mm.free-psutil.virtual_memory().free)
stepsize=np.round(remRam/(afloat*mem_cols)/2,0).astype(int)
ixs=np.arange(0,len(zm)+stepsize,stepsize)
          
for i in range(len(ixs)-1):
    mass[ixs[i]:ixs[i+1]]=((zm[ixs[i]:ixs[i+1]]*mdf.loc[edf.index].values[:mem_cols].T).sum(axis=1)*mass_blowup).round(0).astype(np.uint64)
    # mass[ixs[i]:ixs[i+1]]=(zm[ixs[i]:ixs[i+1]]*mdf.loc[edf.index].values[:mem_cols].T).sum(axis=1)#*mass_blowup)#.round(0).astype(np.uint64)


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

cartesian_time=time.time()-cartesian_time
cartesian_mem=tracemalloc.get_traced_memory()[1]
tracemalloc.reset_peak()



#%% Write unsorted array



if need_batches:

    
    tracemalloc.start() 
    unsorted_time = time.time()
    
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

    #precompute rdbe
    if flag_rdbe_max or flag_rdbe_min:
        
        #base rdbe
        base_rdbe = np.ones(len(zm), dtype=rdbe_bitlim)*2
        if len(Xrdbe[Xrdbe<mem_cols]): base_rdbe +=zm[:, Xrdbe].sum(axis=1)*2
        if len(Yrdbe[Yrdbe<mem_cols]): base_rdbe -=zm[:, Yrdbe].sum(axis=1)
        if len(Zrdbe[Zrdbe<mem_cols]): base_rdbe +=zm[:, Zrdbe].sum(axis=1) 

        #batch rdbe
        batch_rdbeX,batch_rdbeY,batch_rdbeZ=Xrdbe-mem_cols,Yrdbe-mem_cols,Zrdbe-mem_cols
        batch_rdbeX,batch_rdbeY,batch_rdbeZ=batch_rdbeX[batch_rdbeX>-1],batch_rdbeY[batch_rdbeY>-1],batch_rdbeZ[batch_rdbeZ>-1]
        batch_rdbe=np.zeros(len(bm),dtype=rdbe_bitlim)
        if len( batch_rdbeX): batch_rdbe +=bm[:, batch_rdbeX].sum(axis=1)*2
        if len( batch_rdbeY): batch_rdbe -=bm[:, batch_rdbeY].sum(axis=1)
        if len( batch_rdbeZ): batch_rdbe +=bm[:, batch_rdbeZ].sum(axis=1)

    with ExitStack() as stack:
        files = [stack.enter_context(NpyAppendArray(fname, delete_if_exists=True) ) for fname in memfiles]


        for ib, b in enumerate(bm):
            print("writing unsorted batch: "+str(ib)+" ( "+str(np.round((ib+1)/batches*100,2))+" %)")
 
    
            #filter max mass
            q= (mass<=(bmax-am[ib])) #this is tricky since it doesnt know 
            if not q[-1]:#truncate mass
                qtrim=np.argmax(~q)
                zm,mass,base_rdbe=zm[:qtrim],mass[:qtrim],base_rdbe[:qtrim]
            
            zm[:,mem_cols:]=bm[ib]
            
            #find partitions
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
            umparts=np.hstack([0,umparts,len(zm)-1]).astype(int)

            #rdbe filtering
            if flag_rdbe_max or flag_rdbe_min:
                brdbe=base_rdbe+batch_rdbe[ib]
                if flag_rdbe_min & flag_rdbe_max: qr =  (brdbe >= (min_rdbe*2)) & (brdbe <= (max_rdbe*2))
                elif flag_rdbe_min:               qr =  (brdbe >= (min_rdbe*2))
                elif flag_rdbe_max:               qr =  (brdbe <= (max_rdbe*2))    
                
            # write in partitions
            for p in range(partitions):
                l,r=umparts[p],umparts[p+1]
                if r>l:
                    if flag_rdbe_max or flag_rdbe_min: files[p].append(zm[l:r][qr[l:r]])     
                    else:                              files[p].append(zm[l:r])  
                    
                else: files[p].close()
                
                    
                 

                  
  
    print("Completed")
    print(" ")
    
    unsorted_time=time.time()-unsorted_time
    unsorted_mem=tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()
    

 
    #%% Write sorted table
    
    tracemalloc.start() 
    sorted_time = time.time()
    
    zs=np.zeros(bmax+2,dtype=count_bit)
    
    overlap=10 #re-sort overlapping integer roundoff regions
    
    mins,maxs=[],[] #test
    fcomps,fmass=[],[] #placeholder
    with NpyAppendArray(comp_output_path, delete_if_exists=True) as fc:
 
        for p in np.arange(partitions):
            print("partition: "+str(p))
            
            if not os.path.exists(memfiles[p]):
                print("Warning: no compositions found for this partitions!")
                continue
                
                
            comps=np.load(memfiles[p])
            
            #calculate mass (memory efficient batched addition) 
            m = np.zeros(len(comps), dtype=np.uint64)
            remRam=maxRam-(mm.free-psutil.virtual_memory().free)
            stepsize=np.round(remRam/(a64*len(edf))/2,0).astype(int)
            ixs=np.arange(0,len(comps)+stepsize,stepsize)
            for im in range(len(ixs)-1):
                m[ixs[im]:ixs[im+1]]=(np.sum(comps[ixs[im]:ixs[im+1]]*mdf.loc[edf.index].values.T,axis=1)*mass_blowup).round(0).astype(int)

            

            # test            
            # mins.append(m.min()/mass_blowup)
            # maxs.append(m.max()/mass_blowup)
            
            #add to total
            uc=np.bincount(m.astype(np.int64)).astype(count_bit)
            zs[:len(uc)]+=uc
            
            
            if partitions==1:
                fc.append(comps[np.argsort(m,kind="mergesort")])
            
                
            else: #deal with roundoff error in tail    
            
                #get mass tail
                if p<partitions-1: q=m>=len(uc)-overlap-1
    
                #writing
                if p==partitions-1:             fc.append(np.vstack([fcomps,comps])    [np.argsort(np.hstack([fmass,m    ]),kind="mergesort")])
                elif (p>0) & (p<partitions-1):  fc.append(np.vstack([fcomps,comps[~q]])[np.argsort(np.hstack([fmass,m[~q]]),kind="mergesort")])
                elif p==0:                      fc.append(                  comps[~q]  [np.argsort(                 m[~q]  ,kind="mergesort")])
                
                #update previous
                if p<partitions-1: fcomps,fmass=comps[q],m[q]
            
            
            if remove:
                del comps
                os.remove(memfiles[p])
    
    
    sorted_time=time.time()-sorted_time
    sorted_mem=tracemalloc.get_traced_memory()[1]
    
    #Construct mass index
    czs=np.cumsum(zs.astype(np.uint64))
    emp=np.vstack([czs-zs,czs,zs]).T
    np.save(m2g_output_path,emp)
            

#%%
        
if not need_batches:
    
    
    sorted_time = time.time()
    
    np.save(comp_output_path, zm)
    zs=np.bincount(mass.astype(np.int64))

    sorted_time=time.time()-sorted_time
    sorted_mem=tracemalloc.get_traced_memory()[1]
    tracemalloc.reset_peak()

    unsorted_time=0
    unsorted_mem=0


#Construct mass index
cdzs=np.cumsum(zs)
emp=np.vstack([cdzs-zs,cdzs,zs]).T
np.save(m2g_output_path,emp)



#%% test if nonsorted

# test=np.load(comp_output_path,mmap_mode="r")

# #calculate mass (memory efficient batched addition) 

# remRam=maxRam-(mm.free-psutil.virtual_memory().free)
# stepsize=np.round(remRam/(a64*len(edf))/2,0).astype(int)
# ixs=np.arange(0,len(test)+stepsize,stepsize)

# ns=0
# for im in range(len(ixs)-1):
#     l,r=ixs[im],ixs[im+1]
#     if im:
#         l=l-1
    
#     ns+=np.sum(np.diff(((test[l:r]*mdf.loc[edf.index].values.T).sum(axis=1)*mass_blowup).round(0).astype(int))<0)


# print(ns)

# del test
#%% Performance

performance=pd.DataFrame(
[[cartesian_time,cartesian_mem/1e9],
[unsorted_time,unsorted_mem/1e9],
[sorted_time,sorted_mem/1e9]],columns=["time (s)","peak memory (Gigabytes)"])

performance.index=["construct cartesian","construct/write unsorted table","construct/write sorted table"]
performance.loc["total",:]=[performance["time (s)"].sum(),performance["peak memory (Gigabytes)"].max()]

performance.to_csv("performance_space2cart.tsv",sep='\t')

