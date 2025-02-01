# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:06:16 2025

@author: hkleikamp
"""

# clear previous variables 
# Bug, rerunning in spyder IDE causes memory leak, +2 GB each cycle
# from IPython import get_ipython
# try:
#     get_ipython().magic('clear')
#     get_ipython().run_line_magic('reset', '-sf')
#     #get_ipython().magic('reset -sf')
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
# %% change directory to script directory (should work on windows and mac)
basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())
#%% Arguments for execution from Spyder.
#Mass list path
input_file = str(Path(basedir, "test_mass_CASMI2022.txt"))  #: default: CASMI 2022 masses in CartMFP folder default: CartMFP folder/ test_mass_CASMI2022.txt
composition_file= "" #Required: cartesian output file generated by space2cart.py
mass_index_file = composition_file.replace("_comp.npy","_m2g.npy")
#composition arguments (in default mode, these are all derived from the cartesian filename)
#but they can be supplied for post-filtering
composition=""          # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10] 
max_mass = ""           # default 1000
min_rdbe = ""           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
max_rdbe = ""
mode = "pos"                 # ionization mode. Options: " ", "positive", "negative" # positive substracts electron mass, negative adds electron mass, "" doesn't add anything
adducts=["+H+","+Na+","+K+", # default positive adducts 
         "-+","+Cl-"]       # default negative adducts
charges=[1]                            # default: [1]


#performance arguments
ppm = 5                 # ppm for formula prediction
top_candidates = 20     # only save the best predictions sorted by ppm (default 20)
keep_all    = False     # also display mass/ adduct combinations for which no molecular formula was found
#filpaths
mass_table = str(Path(basedir, "mass_table.tsv"))           # table containing element masses, default: CartMFP folder/ mass_table.tsv"
MFP_output_folder = str(Path(basedir, "MFP_Output"))        # default: CartMFP folder / MFP_Output
MFP_output_filename="CartMFP_"+Path(input_file).stem+".tsv" # default: CartMFP_ + input_filename + .tsv 
#%% Arguments for execution from command line.
if not hasattr(sys,'ps1'): #checks if code is executed from command line
    
    import argparse
    parser = argparse.ArgumentParser(
                        prog='CartMFP-cart2form',
                        description='molecular formula prediction, see: https://github.com/hbckleikamp/CartMFP')
    
    #key arguments
    parser.add_argument("-i", "--input_file",       required = False, default=str(Path(basedir, "test_mass_CASMI2022.txt")),
                        help="Required: input mass file, can be txt or tabular format")
    #required yes!
    parser.add_argument("-cart", "--composition_file",       required = True, 
                        help="Required: input cartesian file, as constructed by space2cart.py")
    mass_index_file #derived from composition filename
    
    #output and utility filepaths
    parser.add_argument("-mass_table",                 default=str(Path(basedir, "mass_table.tsv")), required = False, help="list of element masses")  
    parser.add_argument("-mfp_out",  "--MFP_Output  ", default=str(Path(basedir, "MFP_Output")), required = False, help="Output folder for molecular formula prediction")   
    args = parser.parse_args()
    parser.add_argument("-out_file", "--MFP_output_filename",  default="CartMFP_"+Path(args.input_file).stem+".tsv", required = False, help="filename of molecular formula prediction output")   
         
    #composition constraints
    parser.add_argument("-c", "--composition", default="", 
    required = False, help="ALlowed elements and their minimum and maximum count. The following syntax is used: Element_name[minimum_count,maximum_count]")  
    parser.add_argument("-max_mass",  default=0, required = False, help="maximum mass of compositions")  
    parser.add_argument("-min_rdbe",  default=False,   required = False, help="minimum RDBE of compositions. set False to turn off")  
    parser.add_argument("-max_rdbe",  default=False,   required = False, help="maximum RBDE of compositions. set False to turn off")  
    parser.add_argument("-mode",  default="pos",   required = False, help="ionization mode: positive, negative or "" (ignore). This will subtract mass based on ion adducts. if "" is used, the exact masses are used")    
    parser.add_argument("-adducts",  default=["+H+","+Na+","+K+","-+","+Cl-"] ,   
                        required = False, help="The ionization mode will determine used adducts. Syntax: 'sign element charge' eg. gain of H+,Na+,K+ for positive, and  Cl- or loss of H+ for negative ionization mode ")    
    parser.add_argument("-charges",  default=[1],   
                        required = False, help="Charge states considered ")   
    
    #performance arguments
    parser.add_argument("-ppm",  default=5, required = False, help="ppm mass error tolerance of predicted compositions")  
    parser.add_argument("-t","--top_candidates",  default=20, required = False, help="number of best candidates returned (sorted by mass error)")  
    args = parser.parse_args()
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    
    print("")
    print(args) 
    print("")
    locals().update(args)
    

#charges and adducts need command line string parsing
if type(charges)==str: charges=[int(i.strip()) for i in charges.split(",")]
if type(adducts)==str: adducts=[int(i.strip()) for i in adducts.split(",")]
        

#%%
emass = 0.000548579909  # electron mass
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
    #add positive and negative charge mass
    mdf.loc["+",:]=-emass
    mdf.loc["-",:]=emass
    mdf.to_csv("mass_table.tsv", sep="\t")
    

mdf.loc["+"]=-emass
mdf.loc["-"]=+emass


#%% Functions
# read input table (dynamic delimiter detection)
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
                for delim in delims:
                    if delim == " ":
                        delim = "\s"
                    try:
                        tab = pd.read_csv(tabfile, sep=delim, header=None)
                        if Keyword in tab.columns:
                            break
                    except:
                        pass
    return tab
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
    
#%% Check filepaths
if not len(mass_index_file): mass_index_file=composition_file.replace("_comp.npy","_m2g.npy")
if not os.path.exists(input_file):        raise ValueError("Input file  file "+input_file+" not found!")
if not os.path.exists(composition_file):  raise ValueError("Composition file "+composition_file+" not found!")
if not os.path.exists(mass_index_file):   raise ValueError("Mass index file " +mass_index_file +" not found!")

#%% Parse information from composition output
fs=Path(composition_file).stem
mass_blowup=int(fs.split("_b")[-1].split("max")[0])
max_mass=float(fs.split("max")[1].split("rdbe")[0])
bmax=max_mass*mass_blowup
elements=np.array([i.split("]")[-1] for i in fs.split("[")][:-1])
if len(composition):
    #parse composition from filename
    fedf=pd.DataFrame([i.replace("[",",").split(',') for i in  fs.split("]")[:-1]],columns=["symbol","low","high"]).set_index("symbol")
    fedf["low"]=pd.to_numeric(fedf["low"],errors='coerce')
    fedf["high"]=pd.to_numeric(fedf["high"],errors='coerce')
    fedf=fedf.ffill(axis=1)
    fedf[["low","high"]]=fedf[["low","high"]].fillna(0).astype(int)
    #parse composition from composition string
    edf=pd.DataFrame([i.replace("[",",").split(',') for i in  composition.split("]")[:-1]],columns=["symbol","low","high"]).set_index("symbol")
    edf["low"]=pd.to_numeric(edf["low"],errors='coerce')
    edf["high"]=pd.to_numeric(edf["high"],errors='coerce')
    edf=edf.ffill(axis=1)
    
    if edf.isnull().sum().sum(): #fill in missing values from composotion string.
        print("Warning! missing element maxima detected in composition. Imputing from maximum mass (this might affect performance)")
        edf.loc[edf["high"].isnull(),"high"]=(max_mass/mdf.loc[edf.index]).astype(int).values[edf["high"].isnull()].flatten()
    
    edf[["low","high"]]=edf[["low","high"]].fillna(0).astype(int)
    if len(set(edf.index)-set(fedf.index)):  
        print("Warning! composition argument contains elements not present in cartesian file!")
    medf=edf.merge(fedf,on="symbol",how="inner")
    if ((medf["high_y"]-medf["high_x"])<0).any():
        print("Warning! higher element counts observed in allowed composition than is present in cartesian file!")
# !Parse composition string
start_time = time.time()
Xrdbe = np.argwhere(elements == "C").flatten()
Yrdbe = np.argwhere(np.in1d(elements,["H", "F", "Cl", "Br", "I"])).flatten()
Zrdbe = np.argwhere(np.in1d(elements,["N", "P"])).flatten()

#%% read input masses
print("Reading table: "+str(input_file))
print("")
masses = np.unique(read_table(input_file).astype(float).values)
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
    
#%% MFP
start_time=time.time()
print("Loading index files")
print("")
peak_mass = masses*mass_blowup
peak_mass_low = (peak_mass*(1-ppm/1e6)).astype(np.uint64)
peak_mass_high = (peak_mass*(1+ppm/1e6)).astype(np.uint64)
arrs = [np.arange(i, peak_mass_high[ix]+1).astype(np.uint64)
        for ix, i in enumerate(peak_mass_low)]  # integer mass range
a_ix = np.hstack([[ix]*len(i) for ix, i in enumerate(arrs)]) #not really sure about what is going on here
um=np.hstack(arrs)

#filter ppm
ppmd=abs((peak_mass[a_ix].astype(np.int64)-um.astype(np.int64))/peak_mass[a_ix].astype(np.int64)*1e6)<=ppm
um,a_ix=um[ppmd],a_ix[ppmd]
comps = np.load(composition_file, mmap_mode="r")
emp   = np.load(mass_index_file, mmap_mode="r")

#determine RDBE columns on filename
print("Cartesian elapsed time: "+str(time.time()-start_time))
print("")

# Cartesian batched formula prediction
start_time = time.time()
#add batches?
x = emp[um]
q2 = x[:,2]>0
x = x[q2]
q=np.hstack([np.arange(i[0],i[1]) for i in x[:,:2]]).astype(int) #memmap masses
cs= comps[q]
ms=np.sum(cs*mdf.loc[elements].values.reshape(1,-1),axis=1)
us=np.repeat(a_ix[q2],x[:,2].astype(int))
#check if something goes wrong here
ja=np.vstack([np.repeat(um[q2],x[:,2])/mass_blowup,ms]).T #test #-> sorting goes wrong!


# Chemical filtering
flag_rdbe_min = type(min_rdbe) == float or type(min_rdbe) == int
flag_rdbe_max = type(max_rdbe) == float or type(max_rdbe) == int
q=np.ones(len(cs),dtype=bool)
if max_mass: q=q & (ms<=bmax)

#RDBE filtering
rdbe=np.ones(len(cs))#,dtype=np.int8) #dtype !
if len(Xrdbe): rdbe+=np.sum(cs[:,Xrdbe],axis=1) 
if len(Yrdbe): rdbe-=np.sum(cs[:,Yrdbe],axis=1)/2  
if len(Zrdbe): rdbe+=np.sum(cs[:,Zrdbe],axis=1)/2 
            
if   flag_rdbe_min & flag_rdbe_max: q=q & (rdbe>=min_rdbe) &( rdbe<=max_rdbe)
elif flag_rdbe_min:                 q=q & (rdbe>=min_rdbe)
elif flag_rdbe_max:                 q=q & (rdbe<=max_rdbe)

#elemental composition filering
if composition:
    for i in medf.index:
        ev=cs[:,np.argwhere(fedf.index==i)[:,0]].flatten()
        elow,ehigh=fedf.loc[i,"low"],fedf.loc[i,"high"]
        q=q & ev>=elow & ev<=ehigh
ms, cs, us = ms[q], cs[q], us[q]

#ms+=adduct_masses[us] #add adduct mass
print("Picking best "+str(top_candidates)+" candidates.")
pm=peak_mass[us]/mass_blowup 
ppm_diff=(pm-ms)/pm*1e6
#include adduct mass
appm_diff=abs(ppm_diff)
s=np.lexsort((appm_diff,us))
# select top candidates based on ppm
group_ixs=np.argwhere(np.diff(us[s]))[:,0]+1 
s=s[np.hstack([i[:top_candidates] for i in  np.array_split(np.arange(len(s)),group_ixs)])]
pm, ms, cs, ppm_diff = pm[s], ms[s], cs[s], ppm_diff[s]
res=pd.DataFrame(np.hstack([pm.reshape(-1,1),
                            ms.reshape(-1,1),
                            ppm_diff.reshape(-1,1),
                            cs]),columns=["mass","pred_mass","ppm","adduct"]+elements.tolist())
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
MFP_outpath=Path(MFP_output_folder,MFP_output_filename)
res.to_csv(MFP_outpath)
