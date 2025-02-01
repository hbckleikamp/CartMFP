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


# %% change directory to script directory (should work on windows and mac)

basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())

#%% Arguments for execution from Spyder.

#Mass list path
input_file = str(Path(basedir, "test_mass_CASMI2022.txt"))  #: default: CASMI 2022 masses in CartMFP folder default: CartMFP folder/ test_mass_CASMI2022.txt

#composition arguments
composition="H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]"   # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]
max_mass = 1000         # default 1000
min_rdbe = -5           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
max_rdbe = 80

mode = "pos"                 # ionization mode. Options: " ", "positive", "negative" # positive substracts electron mass, negative adds electron mass, "" doesn't add anything
adducts =["--","+H+","+Na+","+K",      # default positive adducts "+H+","+Na+","+K"
         "+-","Cl-"]            # default negative adducts              "+-", "Cl-" 
charges=[1]                            # default: [1]

#performance arguments
ppm = 5                 # ppm for formula prediction
top_candidates = 20     # only save the best predictions sorted by ppm (default 20)
maxmem = 0.7            # fraction of max free memory usage 
mass_blowup = 40000     # converting mass to int (higher blowup -> better precision, but higher memory usage)
keep_all    = False     # also display mass/ adduct combinations for which no molecular formula was found

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
    parser.add_argument("-max_mass",  default=1000, required = False, help="maximum mass of compositions")  
    parser.add_argument("-min_rdbe",  default=-5,   required = False, help="minimum RDBE of compositions. set False to turn off")  
    parser.add_argument("-max_rdbe",  default=80,   required = False, help="maximum RBDE of compositions. set False to turn off")  
    parser.add_argument("-mode", default="pos",   required = False, help="ionization mode: positive, negative or "" (ignore). This will subtract mass based on ion adducts. if "" is used, the exact masses are used")    
    parser.add_argument("-a","--adducts",  default=["+H+","+Na+","+K+","-+","+Cl-"],   
                        required = False, help="The ionization mode will determine used adducts. Syntax: 'sign element charge' eg. gain of H+,Na+,K+ for positive, and  Cl- or loss of H+ for negative ionization mode ")    
    
    parser.add_argument("-charges",  default=[1],   
                        required = False, help="Charge states considered ")    
    
    #performance arguments
    parser.add_argument("-ppm",  default=5, required = False, help="ppm mass error tolerance of predicted compositions")  
    parser.add_argument("-mem",  default=0.7, required = False, help="if <=1: max fraction of available RAM used, if >1: mass RAM usage in GB")  
    parser.add_argument("-t","--top_candidates",  default=20, required = False, help="number of best candidates returned (sorted by mass error)")  
    parser.add_argument("-mass_blowup",  default=40000, required = False, help="multiplication factor to make masses integer. Larger values increase RAM usage but reduce round-off errors")  
    
    
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

# Source: Eli Korvigo (https://stackoverflow.com/questions/28684492/numpy-equivalent-of-itertools-product/28684982)
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
               Keyword=[], #rewrite multiple keywords
               ):

    if type(Keyword)==str: Keyword=[i.strip() for i in Keyword.split(",")]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        try:
            tab = pd.read_excel(tabfile, engine='openpyxl')
        except:
            with open(tabfile, "r") as f:
                tab = pd.DataFrame(f.read().splitlines())

        # dynamic delimiter detection: if file delimiter is different, split using different delimiters until the desired column name is found
        if len(Keyword):
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

    return True,tab


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
if check: masses=np.unique(masses["mz"].astype(float).values)
else:
    check,masses = read_table(input_file,Keyword="mass")
    if check: masses=np.unique(masses["mass"].astype(float).values)
    else: masses=np.unique(pd.read_csv(input_file).iloc[:,-1])

if (masses > max_mass).sum():
    print("masses above maximum mass detected!, filtering masses")
    masses = masses[masses <= max_mass]
print("")
lm=len(masses)

### Add adducts ###
if mode=="":                   adducts=[]
if "n" in mode or "-" in mode: adducts=[a for a in adducts  if a.count("-")==1]
if "p" in mode or "+" in mode: adducts=[a for a in adducts  if a.count("-")!=1]
if (not len(adducts)) & ("n" in mode or "-" in mode): adducts=["+-"]
if (not len(adducts)) & ("p" in mode or "+" in mode): adducts=["--"]


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

if len(mode) & len(adducts): 
    print("adducts used: "+", ".join([i+" ("+str(adduct_mass[ix].round(4)*adduct_sign[ix]) +") " for ix,i in enumerate(adducts)]))
    

if len(adducts):
    i1,i2,i3=np.arange(lm).tolist()*len(adducts), masses.tolist()*len(adducts), np.repeat(np.array(adducts),lm)
    mass_df=pd.DataFrame([i1,i2,i3],index=["index","input_mass","adduct"]).T.merge(adf,on="adduct")
    
else:
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

comp_bitlim=np.uint8
if edf.high.max()>255: 
    comp_bitlim=np.uint16
    print("element cound above 255 detected, using 16bit compositions")
    print("")

m2g_bitlim=np.uint32
bmax=max_mass*mass_blowup
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
onesm = np.array([sys.getsizeof(np.ones(i).astype(np.uint64))
                 for i in dpoints])
a64, b64 = np.linalg.lstsq(np.vstack(
    [dpoints, np.ones(len(dpoints))]).T, onesm.reshape(-1, 1), rcond=None)[0]

rows = 1
memories = []
cols = len(edf)
for i in edf.high:
    rows *= i+1
    s_uint8 = rows*cols*a8[0]+b8[0]
    s_uint64 = rows*a64[0]+b64[0]
    memories.append(s_uint8+s_uint64)

if maxmem<=1: mem_cols = (np.argwhere(np.array(memories) < (mm.free*maxmem))[-1]+1)[0]
if maxmem>1:  mem_cols = (np.argwhere(np.array(memories) <           maxmem)[-1]+1)[0]


need_batches = len(edf)-mem_cols
emem = edf.iloc[:mem_cols]

# construct output path
Cartesian_output_file = "".join(emem.index+"["+emem.low.astype(str)+","+emem.high.astype(
    str)+"]")+"_b"+str(mass_blowup)+"max"+str(int(max_mass))+"rdbe"+str(min_rdbe)+"_"+str(max_rdbe) 

if not len(Cartesian_output_folder):
    Cartesian_output_folder = os.getcwd()
else:
    if not os.path.exists(Cartesian_output_folder):
        os.makedirs(Cartesian_output_folder)

m2g_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_m2g.npy"
comp_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_comp.npy"
mass_output_path = str(
    Path(Cartesian_output_folder, Cartesian_output_file))+"_mass.npy"


print("Output Cartesian file:")
print(Cartesian_output_file)
print("")

# #chemical filtering for RDBE (it is int rounded so pick a generous range)
# #RDBE = X -Y/2+Z/2+1 where X=C Y=H or halogen, Z=N or P https://fiehnlab.ucdavis.edu/projects/seven-golden-rules/ring-double-bonds
Xrdbe = np.argwhere(edf.index == "C").flatten()
Yrdbe = np.argwhere(edf.index.isin(["H", "F", "Cl", "Br", "I"]))[0].flatten()
Zrdbe = np.argwhere(edf.index.isin(["N", "P"])).flatten()

bmax = int(np.round(max_mass*mass_blowup, 0))
if not os.path.exists(m2g_output_path) or not os.path.exists(mass_output_path) or not os.path.exists(comp_output_path) or debug:


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


    if zm.max()<256: comp_bitlim=np.uint8
        

    # add room for remaining columns
    zm = np.hstack(
        [zm, np.zeros([len(zm), len(edf)-mem_cols], dtype=comp_bitlim)])
    s = np.argsort(mass)
    mass, zm = mass[s], zm[s]

    # save and delete to make more memory available
    if not os.path.exists(comp_output_path):
        np.save(comp_output_path, zm)
        
    if not os.path.exists(mass_output_path):
        np.save(mass_output_path, mass)

    del zm

    # create index mappings
    sp = np.argwhere(np.diff(mass) > 0)[:, 0]+1
    six = mass[np.hstack([[0], sp])]

    lbrb=np.hstack([0,np.argwhere(np.diff(mass))[:,0]+1,len(mass)])
    emp = np.zeros([bmax+1,2], dtype=m2g_bitlim) 
    emp[six] = np.vstack([lbrb[:-1],lbrb[1:]]).T
    emp=np.hstack([emp,(emp[:,1]-emp[:,0]).reshape(-1,1)])

    if not os.path.exists(m2g_output_path):
        np.save(m2g_output_path, emp)
        
    del six, sp, s, mass  # or write as function?

else:
    print("Cartesian file found")
    emp = np.load(m2g_output_path)


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
print("")

# %% MFP

print("Loading index files")
print("")
peak_mass = masses*mass_blowup
peak_mass_low = (peak_mass*(1-ppm/1e6)).astype(np.uint64)
peak_mass_high = (peak_mass*(1+ppm/1e6)).astype(np.uint64)
arrs = [np.arange(i, peak_mass_high[ix]+1).astype(np.uint64)
        for ix, i in enumerate(peak_mass_low)]  # integer mass range

a_ix = np.hstack([[ix]*len(i) for ix, i in enumerate(arrs)])
um=np.hstack(arrs)

#filter ppm
ppmd=abs((peak_mass[a_ix].astype(np.int64)-um.astype(np.int64))/peak_mass[a_ix].astype(np.int64)*1e6)<=ppm
um,a_ix=um[ppmd],a_ix[ppmd]


#parse_comp output path


comps = np.load(comp_output_path, mmap_mode="r")
mass  = np.load(mass_output_path, mmap_mode="r")

print("Cartesian elapsed time: "+str(time.time()-start_time))
print("")


# Cartesian batched formula prediction
start_time = time.time()

if need_batches:

    print("MFP batches:")

    am = (bm*edf[mem_cols:].mass.values).sum(axis=1)  # added mass
    us, ms, cs = [], [], []

    for ib, b in enumerate(bm):
        print(ib)
        q1 = um>am[ib]     # non negative mass 
        cm = um-am[ib]     # corrected mass
        cm[~q1]=0
        x = emp[cm]        # get indices of compositions
        q2 = x[:,2]>0      # has composition
        x = x[(q1 & q2)]

        if len(x):
            q=np.hstack([np.arange(i[0],i[1]) for i in x[:,:2]]).astype(int) #memmap masses
            c1= comps[q]
                 
            miss_col=len(edf)-c1.shape[1]
            if miss_col: c1=np.hstack([c1,np.zeros([len(c1),miss_col],dtype=comp_bitlim)])
         
            c1[:, mem_cols:] = b
            cs.append(c1)
            ms.append(mass[q]+am[ib])
            us.append(np.repeat(a_ix[(q1 & q2)],x[:,2].astype(int))) 

            #test
            orig=peak_mass[np.repeat(a_ix[(q1 & q2)],x[:,2].astype(int))].astype(np.int64)
            pred=mass[q]+am[ib].astype(np.int64)
            ppmd=abs((orig-pred)/pred*1e6)
            if (abs((orig-pred)/pred*1e6)>ppm).sum():
                ""+1
            
            

    if len(ms):
        ms, cs, us = np.hstack(ms), np.vstack(cs), np.hstack(us)

else:  # no Cartesian batches
    x = emp[um]
    q2 = x[:,2]>0
    x = x[q2]
    q=np.hstack([np.arange(i[0],i[1]) for i in x[:,:2]]).astype(int) #memmap masses
    cs, ms = comps[q], mass[q]
    us=np.repeat(a_ix[q2],x[:,2].astype(int))


# Chemical filtering
flag_rdbe_min = type(min_rdbe) == float or type(min_rdbe) == int
flag_rdbe_max = type(max_rdbe) == float or type(max_rdbe) == int


if flag_rdbe_min or flag_rdbe_max:
    print("")
    print("RDBE filtering: min: "+str(min_rdbe)+", max: "+str(max_rdbe))

    #make to float
    rdbe = np.ones(len(cs), dtype=np.int8)  # filter on rdbe (int rounded)
    if len(Xrdbe):
        rdbe += cs[:, Xrdbe][:, 0]
    if len(Yrdbe):
        rdbe -= (cs[:, Yrdbe][:, 0]/2).astype(int) # not fully accurate because of int roundoff
    if len(Zrdbe):
        rdbe += (cs[:, Zrdbe][:, 0]/2).astype(int) # not fully accurate because of int roundoff

    if flag_rdbe_min & flag_rdbe_max:
        q = (rdbe >= min_rdbe) & (rdbe <= max_rdbe)
    elif flag_rdbe_min:
        q = (rdbe >= min_rdbe)
    elif flag_rdbe_max:
        q = (rdbe <= max_rdbe)

    ms, cs, us = ms[q], cs[q], us[q]


print("Picking best "+str(top_candidates)+" candidates.")

pm=peak_mass[us] 
ppm_diff=(pm.astype(np.int64)-ms.astype(np.int64))/pm.astype(np.int64)*1e6

appm_diff=abs(ppm_diff)
s=np.lexsort((appm_diff,us))

# select top candidates based on ppm
group_ixs=np.argwhere(np.diff(us[s]))[:,0]+1 
s=s[np.hstack([i[:top_candidates] for i in  np.array_split(np.arange(len(s)),group_ixs)])]
pm, ms, cs, ppm_diff = pm[s], ms[s], cs[s], ppm_diff[s]
res=pd.DataFrame(np.hstack([pm.reshape(-1,1)/mass_blowup,
                            ms.reshape(-1,1)/mass_blowup,
                            ppm_diff.reshape(-1,1),
                            cs]),columns=["mass","pred_mass","ppm"]+elements.tolist())




print("")
print("MFP elapsed time: "+str(time.time()-start_time))

res=pd.concat([mass_df.iloc[us[s],:].reset_index(drop=True),res[["pred_mass","ppm"]+elements.tolist()]],axis=1)

#add adduct formulas
res[list(set(acomps.columns)-set(elements))]=0
res[acomps.columns]+=acomps.loc[res.adduct,acomps.columns].values

if keep_all: #add unidentified masses
    missing_index=list(set(mass_df["index"])-set(res.index))
    missing_rows=mass_df[["index","input_mass"]].drop_duplicates().set_index("index").loc[missing_index,:].reset_index()
    missing_rows[res.columns[2:]]=0
    missing_rows["adduct"]=""
    res=pd.concat([res,missing_rows])
    
res["appm"]=res["ppm"].abs()
res=res.sort_values(by=["index","charge","appm"]).reset_index(drop=True)   

#%%
if not os.path.exists(MFP_output_folder): os.makedirs(MFP_output_folder)
if not len(MFP_output_filename): MFP_output_filename="CartMFP_"+Path(input_file).stem+".tsv"
MFP_outpath=str(Path(MFP_output_folder,MFP_output_filename))
print("Writing output: "+MFP_outpath)
res.to_csv(MFP_outpath)
