# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:35:44 2025

@author: hkleikamp
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:54:59 2024

@author: hkleikamp
"""



#%% clear variables and console

# try:
#     from IPython import get_ipython
#     get_ipython().magic('clear')
#     get_ipython().magic('reset -f')
# except:
#     pass


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
import pyarrow
import pyarrow.csv as pc

# %% change directory to script directory (should work on windows and mac)

basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())

#%% Arguments for execution from Spyder.

#Mass list path
input_file = str(Path(basedir, "test_mass_CASMI2022.txt"))  #: default: CASMI 2022 masses in CartMFP folder default: CartMFP folder/ test_mass_CASMI2022.txt

#composition arguments
composition="H[200]C[75]N[50]O[50]P[10]S[10]"   # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]
max_mass = 1000         # default 1000
min_rdbe = -5           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
max_rdbe = 80           

mode = "pos"                    # ionization mode. Options: "", "positive", "negative" # positive substracts electron mass, negative adds electron mass, "" doesn't add anything
adducts =["--","+Na+","+K",     # default positive adducts "+H+","+Na+","+K"
         "+-","Cl-"]            # default negative adducts              "+-", "Cl-" 
charges=[1]                            # default: [1]

#performance arguments
ppm = 5                   # ppm for formula prediction
top_candidates = 20       # only save the best predictions sorted by ppm (default 20)
maxmem = 10e9 #0.7              # fraction of max free memory usage 
mass_blowup = 40000     # converting mass to int (higher blowup -> better precision, but higher memory usage)
keep_all    = True       # also display mass/ adduct combinations for which no molecular formula was found

#filpaths
mass_table = str(Path(basedir, "mass_table.tsv"))           # table containing element masses, default: CartMFP folder/ mass_table.tsv"
Cartesian_output_folder = str(Path(basedir, "Cart_Output")) # default: CartMFP folder / Cart_Output
MFP_output_folder = str(Path(basedir, "MFP_Output"))        # default: CartMFP folder / MFP_Output
MFP_output_filename=""                                      # default: CartMFP_ + input_filename + .tsv 


debug=False #True     #writes CART MFP file even if it already exists to test writing function


#%% Arguments for execution from command line.

if not hasattr(sys,'ps1'): #checks if code is executed from command line
    
    import argparse

    parser = argparse.ArgumentParser(
                        prog='CartMFP-mass2form',
                        description='molecular formula prediction, see: https://github.com/hbckleikamp/CartMFP')
    
    #key argument
    parser.add_argument("-i", "--input_file",       required = False, default=str(Path(basedir, "test_mass_CASMI2022.txt")),
                        help="Required: input mass file, can be txt or tabular format")
    
    #output and utility filepaths
    parser.add_argument("-mass_table",                 default=str(Path(basedir, "mass_table.tsv")), required = False, help="list of element masses")  
    parser.add_argument("-cart_out", "--Cartesian_output_folder",  default=str(Path(basedir, "Cart_Output")), required = False, help="Output folder for cartesian files")   
    parser.add_argument("-mfp_out",  "--MFP_output_folder", default=str(Path(basedir, "MFP_Output")), required = False, help="Output folder for molecular formula prediction")   
    parser.add_argument("-out_file", "--MFP_output_filename",  default="", required = False, help="filename of molecular formula prediction output")   
     
    #composition constraints
    parser.add_argument("-c", "--composition", default="H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]", 
    required = False, help="ALlowed elements and their minimum and maximum count. The following syntax is used: Element_name[minimum_count,maximum_count]")  
    parser.add_argument("-max_mass",  default=1000, required = False, help="maximum mass of compositions",type=float)  
    parser.add_argument("-min_rdbe",  default=-5,   required = False, help="minimum RDBE of compositions. set False to turn off",type=float)  
    parser.add_argument("-max_rdbe",  default=80,   required = False, help="maximum RBDE of compositions. set False to turn off",type=float)  
    parser.add_argument("-mode", default="pos",   required = False, help="ionization mode: positive, negative or "" (ignore). This will subtract mass based on ion adducts. if "" is used, the exact masses are used")    
    parser.add_argument("-a","--adducts",  default=["+H+","+Na+","+K+","-+","+Cl-"],   
                        required = False, help="The ionization mode will determine used adducts. Syntax: 'sign element charge' eg. gain of H+,Na+,K+ for positive, and  Cl- or loss of H+ for negative ionization mode ")    
    
    parser.add_argument("-charges",  default=[1],   
                        required = False, help="Charge states considered ")    
    
    #performance arguments
    parser.add_argument("-ppm",  default=5, required = False, help="ppm mass error tolerance of predicted compositions",type=float)  
    parser.add_argument("-mem",  default=0.7, required = False, help="if <=1: max fraction of available RAM used, if >1: mass RAM usage in GB",type=float)  
    parser.add_argument("-t","--top_candidates",  default=20, required = False, help="number of best candidates returned (sorted by mass error)",type=int)  
    parser.add_argument("-keep_all",  default=False, required = False, help="keep masses with no predicted formula in output")  
    parser.add_argument("-mass_blowup",  default=40000, required = False, help="multiplication factor to make masses integer. Larger values increase RAM usage but reduce round-off errors",type=int)  
    
    
    parser.add_argument("-d","--debug",  default=False, required = False, help="")  
    

    args = parser.parse_args()
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    
    print("")
    print(args) 
    print("")
    locals().update(args)

#charges and adducts need command line string parsing
if type(charges)==str: charges=[int(i.strip()) for i in charges.split(",")]
if type(adducts)==str: adducts=[i.strip() for i in adducts.split(",")]

#%%
emass = 0.000548579909  # electron mass


# %% General functions

# Source: pv (https://stackoverflow.com/questions/28684492/numpy-equivalent-of-itertools-product/28684982)
def cartesian(arrays, comp_bitlim=np.uint8, out=None):
    n = reduce(operator.mul, [x.size for x in arrays], 1)
    print(n)
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=comp_bitlim)

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
    

#https://stackoverflow.com/questions/41833740/numpy-index-of-the-maximum-with-reduction-numpy-argmax-reduceat
def numpy_argmin_reduceat(a,b):
    n = a.max()+1  # limit-offset
    grp_count = np.append(b[1:] - b[:-1], a.size - b[-1])
    shift = n*np.repeat(np.arange(grp_count.size), grp_count)
    return (a+shift).argsort()[b]

#https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
def create_ranges(a):
    l = a[:,1] - a[:,0]
    clens = l.cumsum()
    ids = np.ones(clens[-1],dtype=int)
    ids[0] = a[0,0]
    ids[clens[:-1]] = a[1:,0] - a[:-1,1]+1
    return ids.cumsum()


#The number of bits needed to represent an integer n is given by rounding down log2(n) and then adding 1
def bits(x,neg=False):
    bitsize=np.array([8,16,32,64])
    dtypes=[np.uint8,np.uint16,np.uint32,np.uint64]
    if neg: dtypes=[np.int8,np.int16,np.int32,np.int64]
    return dtypes[np.argwhere(bitsize-(np.log2(x)+1)>0).min()]


# %% Get elemental metadata 
#(this is only executed when the mass table file is not found)

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

mdf.loc["+"]=-emass
mdf.loc["-"]=emass

#%% read input masses

print("Reading table: "+str(input_file))
print("")


#read masses
check,masses = read_table(input_file,Keyword="mz")
mass_ix=np.arange(len(masses))
if check: masses,umass_ix=np.unique(masses["mz"].astype(float).values,return_inverse=True) 
else:
    check,masses = read_table(input_file,Keyword="mass")
    if check: masses,umass_ix=np.unique(masses["mass"].astype(float).values,return_inverse=True)
    else: masses,umass_ix=np.unique(pd.read_csv(input_file).iloc[:,-1],return_inverse=True)
    
map_umass=pd.DataFrame(np.vstack([mass_ix,umass_ix]).T,columns=["original_index","index"])
    

if (masses > max_mass).sum():
    print("masses above maximum mass detected!, filtering masses")
    masses = masses[masses <= max_mass]
print("")
lm=len(masses)

### Add adducts ###
if   "n" in mode or "-" in mode: adducts=[a for a in adducts  if a.count("-")==1]
elif "p" in mode or "+" in mode: adducts=[a for a in adducts  if a.count("-")!=1]
else: adducts=[]
if (not len(adducts)) & ("n" in mode or "-" in mode): adducts=["+-"]
if (not len(adducts)) & ("p" in mode or "+" in mode): adducts=["--"]

if len(adducts):
    adduct_sign    =[-1  if a[0]=="-" else  1  for a in adducts]
    adduct_mass=[getMz(a[1:]) for a in adducts]
    
    adf=pd.DataFrame([adducts,adduct_mass]).T
    adf.columns=["adduct","adduct_mass"]
    adf["adduct_mass"]=adf["adduct_mass"]*adduct_sign
    acomps=pd.concat([parse_form(a[1:]) for a in adducts]).fillna(0)*np.array(adduct_sign).reshape(-1,1)
    [acomps.pop(i) for i in ["+","-"] if i in acomps.columns]
    acomps.index=adducts
    # adf[acomps.columns]=(acomps*np.array(adduct_sign).reshape(-1,1)).values
    
    if "n" in mode or "-" in mode:  print("mode is negative,")
    if "p" in mode or "+" in mode:  print("mode is positive,")
    print("adducts used: "+", ".join([i+" ("+str(adduct_mass[ix].round(4)*adduct_sign[ix]) +") " for ix,i in enumerate(adducts)]))
        
    i1,i2,i3=np.arange(lm).tolist()*len(adducts), masses.tolist()*len(adducts), np.repeat(np.array(adducts),lm)
    mass_df=pd.DataFrame([i1,i2,i3],index=["index","input_mass","adduct"]).T.merge(adf,on="adduct")
    
else:
    print("no adducts used!")
    
    mass_df=pd.DataFrame([np.arange(lm),masses],index=["index","input_mass"]).T
    mass_df["adduct_mass"]=0
    mass_df["adduct"]=""
    

### Add charges ###
mass_df["charge"]=[charges]*len(mass_df)
mass_df=mass_df.explode("charge")
mass_df["mass"]=mass_df["input_mass"]*mass_df["charge"]-mass_df["adduct_mass"]*mass_df["charge"]

mass_df=mass_df[mass_df["mass"]<max_mass].reset_index(drop=True) #filter on max mass

adduct_cats=mass_df["adduct"].values.flatten()
charge_cats=mass_df["charge"].values
masses=mass_df["mass"].values

bmax = int(np.round((max_mass+1)*mass_blowup, 0))
peak_mass = masses*mass_blowup
pmi=peak_mass.astype(int)
peak_mass_low = np.clip((peak_mass*(1-ppm/1e6)),0,bmax).astype(np.int64)
peak_mass_high = np.clip((peak_mass*(1+ppm/1e6)),0,bmax).astype(np.int64)

d=(peak_mass_high-peak_mass_low)

#masses to search
um=(np.repeat(peak_mass_low,d)+(np.arange(d.sum()) - np.repeat(np.cumsum(d)-d, d))).astype(np.uint64)

#index to link back to original mass
a_ix=np.repeat(np.arange(len(masses)),d)

# %% Construct MFP space

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

comp_bitlim=np.uint8
if edf.high.max()>255: 
    comp_bitlim=np.uint16
    print("element cound above 255 detected, using 16bit compositions")
    print("")

m2g_bitlim=np.uint32
# bmax=max_mass*mass_blowup
if bmax>=4294967296:
   m2g_bitlim=np.uint64 

# % Determine number of element batches
mm = psutil.virtual_memory()
dpoints = np.array([10, 100, 1e3, 1e4, 1e5, 1e6]).astype(int)

# size of uint8 array
onesm = np.array([sys.getsizeof(np.ones(i).astype(comp_bitlim)) for i in dpoints])
a8, b8 = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

# size of uint64 array
onesm = np.array([sys.getsizeof(np.ones(i).astype(np.uint64)) for i in dpoints])
a64, b64 = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

#size of float array
onesm = np.array([sys.getsizeof((np.ones(i)).astype(np.float64)) for i in dpoints])
afloat, bfloat = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

rows = 1
memories = []
cols = len(edf)
for i in edf.arr:
    rows *= len(i)
    s_uint8 = rows*cols*a8[0]+b8[0]
    s_uint64 = rows*a64[0]+b64[0]*2
    memories.append(s_uint8+s_uint64)
 
if maxmem<1: maxRam=mm.free*maxmem
else:        maxRam=maxmem
mem_cols = (np.argwhere(np.array(memories) < (maxRam))[-1]+1)[0]


need_batches = len(edf)-mem_cols
emem = edf.iloc[:mem_cols]

# construct output path
Cartesian_output_file = "".join(emem.index+"["+emem.low.astype(str)+","+emem.high.astype(
    str)+"]")+"_b"+str(mass_blowup)+"max"+str(int(max_mass))+"rdbe"+str(min_rdbe)+"_"+str(max_rdbe) 
Cartesian_output_file=Cartesian_output_file.replace("[0,","[")

if not len(Cartesian_output_folder):
    Cartesian_output_folder = os.getcwd()
else:
    if not os.path.exists(Cartesian_output_folder):
        os.makedirs(Cartesian_output_folder)

m2g_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_m2g.npy"
comp_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_comp.npy"


print("Output Cartesian file:")
print(Cartesian_output_file)
print("")

# #chemical filtering for RDBE (it is int rounded so pick a generous range)
# #RDBE = X -Y/2+Z/2+1 where X=C Y=H or halogen, Z=N or P https://fiehnlab.ucdavis.edu/projects/seven-golden-rules/ring-double-bonds
flag_rdbe_min = type(min_rdbe) == float or type(min_rdbe) == int
flag_rdbe_max = type(max_rdbe) == float or type(max_rdbe) == int
rdbe_bitlim=np.int16 #np.int8 causes int sign overflow


Xrdbe = np.argwhere(edf.index == "C").flatten()
Yrdbe = np.argwhere(edf.index.isin(["H", "F", "Cl", "Br", "I"])).flatten()
Zrdbe = np.argwhere(edf.index.isin(["N", "P"])).flatten()


# compute cartesian batches
print("")
bm=[]
if need_batches:

    batches = reduce(operator.mul, edf.iloc[mem_cols:]["high"].values+1, 1)
    print("array too large for memory, performing cartesian product in batches: "+str(batches))

    # compute cartesian product of the remaining elements
    arrays = edf.arr.values[mem_cols:].tolist()
    print("")
    print("computing remaining cartesian:")
    bm = cartesian(arrays)
    am = ((bm*mdf.loc[edf.index].values[mem_cols:].reshape(1,-1)).sum(axis=1)*mass_blowup).round(0).astype(np.int64) 
print("")



if not os.path.exists(m2g_output_path) or not os.path.exists(comp_output_path) or debug:


    # % Compute base cartesian
    print("constructing base cartesian:")
    arrays = edf.arr.values[:mem_cols].tolist()
    arrays=[i.astype(comp_bitlim) for i in arrays]

    zm = cartesian(arrays)

    # batched addition
    mass = np.zeros(len(zm), dtype=np.uint64)
    #memory efficient batched addition 
    remRam=maxRam-(mm.free-psutil.virtual_memory().free)
    stepsize=np.round(remRam/(afloat*mem_cols)/2,0).astype(int)
    ixs=np.arange(0,len(zm)+stepsize,stepsize)
           
    for i in range(len(ixs)-1):
        mass[ixs[i]:ixs[i+1]]=((zm[ixs[i]:ixs[i+1]]*mdf.loc[edf.index].values[:mem_cols].T).sum(axis=1)*mass_blowup).round(0).astype(np.uint64)

    # filter base cartesian on maximum mass
    if max_mass:  # (recommended!)
        zm = zm[mass <= bmax]
        mass = mass[mass <= bmax]


    if zm.max()<256: comp_bitlim=np.uint8
    
    # add room for remaining columns
    zm = np.hstack(
        [zm, np.zeros([len(zm), len(edf)-mem_cols], dtype=comp_bitlim)])
    s = np.argsort(mass)
    mass, zm = mass[s], zm[s]
    
    # save and delete to make more memory available
    if os.path.exists(comp_output_path): os.remove(comp_output_path)
    np.save(comp_output_path, zm)
    
    del zm

    # create index mappings
    zs=np.bincount(mass.astype(np.int64))
    count_bit=bits(zs.max()*len(bm))
    zs=zs.astype(count_bit)
    
    cdzs=np.cumsum(zs)
    emp=np.vstack([cdzs-zs,cdzs,zs]).T
    

    if os.path.exists(m2g_output_path): os.remove(m2g_output_path)
    np.save(m2g_output_path, emp)
    
    del emp,zs,cdzs,s, mass  # or write as function?


# %% MFP

print("Loading index files")
print("")


#parse_comp output path
comps = np.load(comp_output_path, mmap_mode="r")
emp = np.load(m2g_output_path, mmap_mode="r")
mass=np.repeat(np.arange(len(emp)),emp[:,2])

#calculate base rdbe
if flag_rdbe_max or flag_rdbe_min:
    rdbe = np.ones(len(comps), dtype=rdbe_bitlim)*2
    if len(Xrdbe[Xrdbe<mem_cols]): rdbe =rdbe+comps[:, Xrdbe].sum(axis=1)*2
    if len(Yrdbe[Yrdbe<mem_cols]): rdbe =rdbe-comps[:, Yrdbe].sum(axis=1)
    if len(Zrdbe[Zrdbe<mem_cols]): rdbe =rdbe+comps[:, Zrdbe].sum(axis=1) 


    batch_rdbeX,batch_rdbeY,batch_rdbeZ=Xrdbe-mem_cols,Yrdbe-mem_cols,Zrdbe-mem_cols
    batch_rdbeX,batch_rdbeY,batch_rdbeZ=batch_rdbeX[batch_rdbeX>-1],batch_rdbeY[batch_rdbeY>-1],batch_rdbeZ[batch_rdbeZ>-1]
    batch_rdbe=np.zeros(len(bm),dtype=rdbe_bitlim)
    if len( batch_rdbeX): batch_rdbe =batch_rdbe+bm[:, batch_rdbeX].sum(axis=1)*2
    if len( batch_rdbeY): batch_rdbe =batch_rdbe-bm[:, batch_rdbeY].sum(axis=1)
    if len( batch_rdbeZ): batch_rdbe =batch_rdbe+bm[:, batch_rdbeZ].sum(axis=1)

#%%
cartesian_time=time.time()-cartesian_time


print("Cartesian elapsed time: "+str(cartesian_time))
print("")

# Cartesian batched formula prediction
mfp_time = time.time()


#%%
total_filtering_time=0
max_filtering_mem=[]
total_pick_best_time=0
max_pick_best_mem=[]

ibits=bits(len(masses))
if need_batches:

    print("MFP batches: "+str(len(bm)))


    us, cs, ds = [], [], []
    for ib, b in enumerate(bm):
        print(ib)
        cm = um-am[ib]     # corrected input mass
        q = um>=am[ib]         # non negative mass
        qa_ix,cm=a_ix[q],cm[q] # filter q
        x = emp[cm]            # get indices of compositions

        #get mass indices 
        mr=np.arange(x[:,2].sum()) - np.repeat(np.cumsum(x[:,2])-x[:,2], x[:,2]) #make extended range from emp cumsum
        q=np.repeat(x[:,0],x[:,2])+mr #composition index
        ea_ix=np.repeat(qa_ix,x[:,2].astype(ibits))


        ### Chemical filtering
        if flag_rdbe_max or flag_rdbe_min:
            brdbe=rdbe[q]+batch_rdbe[ib]
            if flag_rdbe_min & flag_rdbe_max: qr = (brdbe >= (min_rdbe*2)) & (brdbe <= (max_rdbe*2))
            elif flag_rdbe_min:               qr = (brdbe >= (min_rdbe*2))
            elif flag_rdbe_max:               qr = (brdbe <= (max_rdbe*2))
            q,ea_ix=q[qr],ea_ix[qr]
            

        if not len(q): continue

        ### Pre-trim best candidates

        
        #pick best per batch
        ms=mass[q]+am[ib]
        d=abs(ms-pmi[ea_ix]).astype(np.int16)
        group_ixs=np.hstack([0,np.argwhere(ea_ix[1:]>ea_ix[:-1])[:,0]+1]) 
        lows=numpy_argmin_reduceat(d,group_ixs) 
        ls=np.clip(lows-top_candidates,group_ixs,None)
        rs=np.clip(lows+top_candidates,ls+1,np.hstack([group_ixs[1:],len(d)]))
        xtrim=create_ranges(np.vstack([ls,rs]).T)

        #MFP
        c1=comps[q[xtrim]]
        miss_col=len(edf)-c1.shape[1]
        if miss_col:
            print("Warning! Cartesian table has different shape!")
            c1=np.hstack([c1,np.zeros([len(c1),miss_col],dtype=comp_bitlim)])
        c1[:, mem_cols:] = b
        
        
        cs.append(c1)
        us.append(ea_ix[xtrim]) 
        ds.append(d[xtrim])

    if len(cs):
        cs, us, ds =  np.vstack(cs), np.hstack(us), np.hstack(ds)


else:  # no Cartesian batches
    x = emp[um]
    q2 = x[:,2]>0
    x = x[q2]
    mr=np.arange(x[:,2].sum()) - np.repeat(np.cumsum(x[:,2])-x[:,2], x[:,2])
    q=np.repeat(x[:,0],x[:,2])+mr
    cs= comps[q]
    us=np.repeat(a_ix[q2],x[:,2].astype(int))

mfp_time=time.time()-mfp_time-total_filtering_time-total_pick_best_time



#%% Pick best candidates


print("Picking best "+str(top_candidates)+" candidates.")


#pick best per mass
s=np.lexsort((ds,us)) #maybe a faster solution than lexsort exists?
ds=ds[s]
group_ixs=np.argwhere(us[s[1:]]!=us[s[:-1]])[:,0]+1
max_mass=ds[np.hstack([0,group_ixs])+top_candidates]+1 #+1 for roundoff error
q=s[np.argwhere((ds-np.repeat(max_mass,np.diff(np.hstack([0,group_ixs,len(ds)]))))<0)[:,0]]
cs,us=cs[q],us[q]


#pick best per input mass within ppm
print("")
print("MFP elapsed time: "+str(mfp_time))

res=pd.DataFrame(mass_df.iloc[us,:])
res["pred_mass"]=np.sum(cs*mdf.loc[edf.index].values.flatten(),axis=1)
res["ppm"]=(res["pred_mass"]-res["mass"])/res["mass"]*1e6
res["appm"]=res["ppm"].abs()
res["rdbe"]=(cs[:, Xrdbe].sum(axis=1).astype(rdbe_bitlim)*2
             -cs[:, Yrdbe].sum(axis=1).astype(rdbe_bitlim)
             +cs[:, Zrdbe].sum(axis=1).astype(rdbe_bitlim))/2+1

res[elements]=cs
res=res[res["appm"]<=ppm]
res=res.sort_values(by=["index","charge","appm"]).reset_index(drop=True)   
res=map_umass.merge(res,on="index",how="inner") #map back non-unique mass index
res=res.groupby("original_index",sort=False).head(top_candidates) #final pick best


#%% combine with original index


if len(adducts): 
    res[list(set(acomps.columns)-set(elements))]=0
    
q=res.columns.isin(mdf.index)
hill=res.columns[q].sort_values().tolist()
res=res[res.columns[~q].tolist()+hill]


#construct element string
ecounts=res[hill]
e_arr=np.tile(hill,len(res)).reshape(len(res),-1) #np.array([hill]*len(res)) #slow
e_arr=np.where(ecounts==0,"",e_arr)
eles=ecounts.applymap(str).replace("0","").replace("1","")
res["formula"]=["".join(i) for i in np.hstack([e_arr,eles])[:,np.repeat(np.arange(len(hill)),2)+np.tile(np.array([0,len(hill)]),len(hill))]]

#generate element string with adducts
if len(adducts): 
    res[acomps.columns]+=acomps.loc[res.adduct,acomps.columns].values
    ecounts=res[hill]
    e_arr=np.tile(hill,len(res)).reshape(len(res),-1) #np.array([hill]*len(res)) #slow
    e_arr=np.where(ecounts==0,"",e_arr)
    eles=ecounts.applymap(str).replace("0","").replace("1","")
    res["formula+adduct"]=["".join(i) for i in np.hstack([e_arr,eles])[:,np.repeat(np.arange(len(hill)),2)+np.tile(np.array([0,len(hill)]),len(hill))]]




if keep_all: 
    missing_index=np.argwhere(~np.in1d(np.arange(lm),np.unique(us))).tolist()
    
    if len(missing_index):
        missing_rows=mass_df[["index","input_mass"]].drop_duplicates().set_index("index").loc[missing_index,:].reset_index()
        missing_rows[res.columns[2:]]=0
        missing_rows["adduct"]=""
        res=pd.concat([res,missing_rows])


#%% Write

if not os.path.exists(MFP_output_folder): os.makedirs(MFP_output_folder)
if not len(MFP_output_filename): MFP_output_filename="CartMFP_"+Path(input_file).stem+".tsv"
MFP_outpath=str(Path(MFP_output_folder,MFP_output_filename))

#faster writing than pandas
new_pa_dataframe = pyarrow.Table.from_pandas(res)
write_options = pc.WriteOptions(delimiter="\t",batch_size=10000)
pc.write_csv(new_pa_dataframe, MFP_outpath,write_options)
