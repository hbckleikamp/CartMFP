# CartMFP

CartMFP or Cartesian molecular formula prediction, is a python tool that can perform molecular formula predictions on custom databases.

<br>
#### How does CartMFP work

CartMFP consists of two steps.
1. Constructing a database *space2cart.py* : A local database is constructed. <br>
2. Molecular formula precition *cart2form.py* : Molecular formulas that are predicted based on input masses. <br>

<br>
#### 1. Constructing a database

A database is constructed by enumerating all combinations of elements within a certain range.
The compositional space is described with a specific syntax: Element[min,max].
This can be any element in the periodic table for which the monoisotopic mass is described in the NIST database.
"H[200]C[75]N[50]O[50]P[10]S[10]" is used default elemental space.
This would eqaute to 0-200 Hydrogen, 0-75 Carbon, 0-50 Nitrogen, etc. 

Apart from max element constraints, the elemental composition space is further limited by the maximum mass `max_mass` and ring double bond equivalents (RDBE) `min_rdbe`,`max_rdbe`.

Base chemical constraints:
|Parameter           | Deaulft value     |       Description|
|-----------------|:-----------:|---------------|
|composition| "H[200]C[75]N[50]O[50]P[10]S[10]" | composition string describing minimum and maximum element counts|
|max_mass| 1000 | maximum mass (Da)|
|min_rdbe | -5 | minimum RDBE |
|max_rdbe| 80 | maxmimum RDBE |


Advanced chemical constraints are provided by implementing some of Fiehn's 7 Golden rules.
Which filters unrealistic or impossible compositions.
This can drastically reduce the size of your composition space.

|Parameter           | Deaulft value     |       Description|
|-----------------|:-----------:|---------------|
|composition| "H[200]C[75]N[50]O[50]P[10]S[10]" | composition string describing minimum and maximum element counts|
|max_mass| 1000 | maximum mass (Da)|
|min_rdbe | -5 | minimum RDBE |
|max_rdbe| 80 | maxmimum RDBE |

#advanced chemical rules
filt_7gr=True                                                                       #Toggles all advanced chemical filtering using rules #2,4,5,6 of Fiehn's "7 Golden Rules" 
filt_LewisSenior=True                                                               #Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence)
filt_ratios="HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]"  #Golden Rules #4,5: Filter on chemical ratios with extended range 99.9% coverage
filt_NOPS=True   




parameters:



#### Installation

#pip

#### What outputs does it generate? 

NovoLign generates several output files divided over different folders.
|Folder           | Section     |       Contents|
|-----------------|:-----------:|---------------|
|diamond_fasta| 1 | generated fasta file from input peptide sequences file for DIAMOND alignment|
|diamond_alignments| 1 | DIAMOND alignment|
|lca| 2 | different LCA outputs: conventional (CON) weighted (W), bitscore weighted (BIT) |
|composition| 3 | taxonomic composition of input sample (for different LCAs) including level of decoy matches|
|experiment_qc| 4 | comparison of spectral annotation rates of input sample by NovoLign and database searching, for spectra at different quality levels|
|database_qc| 5 | comparison of taxonomic composition of input sample obtained by NovoLign to taxonomic composition obtained by database searching |
|psms| 5 | Final PSMs format output with NovoLign annotation |

<br>


### Execute CartMFP from command line ###


Basic usage 
```
python "mass2form.py" -i "test_mass_CASMI2022.txt"
```

With database custom element space (default= "H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]" ):
```
python "mass2form.py" -i "test_mass_CASMI2022.txt" -c "H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]"
```

#### Licensing

The pipeline is licensed with standard MIT-license. <br>
If you would like to use this pipeline in your research, please cite the following papers: 
      



#### Contact:
-Hugo Kleimamp (Developer): hugo.kleikamp@uantwerpen.be<br> 

