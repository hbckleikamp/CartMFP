# CartMFP

CartMFP or Cartesian molecular formula prediction, is a python tool that can perform molecular formula predictions on custom databases.

#### How does CartMFP work

CartMFP consists of two steps.
1. Constructing a database *space2cart.py* : A local database is constructed. <br>
2. Molecular formula precition *cart2form.py* : Molecular formulas that are predicted based on input masses. <br>

#### 1. Constructing a database

A database is constructed by enumerating all combinations of elements within a certain range.
The compositional space is described with a specific syntax: Element[min,max].
This can be any element in the periodic table for which the monoisotopic mass is described in the NIST database.
"H[200]C[75]N[50]O[50]P[10]S[10]" is used default elemental space.
This would eqaute to 0-200 Hydrogen, 0-75 




NovoLign generates several output files divided over different folders.
|Parameter           | Decaulft     |       Description|
|-----------------|:-----------:|---------------|
|diamond_fasta| 1 | generated fasta file from input peptide sequences file for DIAMOND alignment|
|diamond_alignments| 1 | DIAMOND alignment|
|lca| 2 | different LCA outputs: conventional (CON) weighted (W), bitscore weighted (BIT) |
|composition| 3 | taxonomic composition of input sample (for different LCAs) including level of decoy matches|
|experiment_qc| 4 | comparison of spectral annotation rates of input sample by NovoLign and database searching, for spectra at different quality levels|
|database_qc| 5 | comparison of taxonomic composition of input sample obtained by NovoLign to taxonomic composition obtained by database searching |
|psms| 5 | Final PSMs format output with NovoLign annotation |


#composition arguments
composition="H[200]C[75]N[50]O[50]P[10]S[10]"   # default: H[0,200]C[0,75]N[0,50]O[0,50]P[0,10]S[0,10]
max_mass = 1000         # default 1000
min_rdbe = -5           # rdbe filtering default range -5,80 (max rdbe will depend on max mass range)
max_rdbe = 80


|Folder           | Section     |       Contents|
|-----------------|:-----------:|---------------|
|diamond_fasta| 1 | generated fasta file from input peptide sequences file for DIAMOND alignment|
|diamond_alignments| 1 | DIAMOND alignment|
|lca| 2 | different LCA outputs: conventional (CON) weighted (W), bitscore weighted (BIT) |
|composition| 3 | taxonomic composition of input sample (for different LCAs) including level of decoy matches|
|experiment_qc| 4 | comparison of spectral annotation rates of input sample by NovoLign and database searching, for spectra at different quality levels|
|database_qc| 5 | comparison of taxonomic composition of input sample obtained by NovoLign to taxonomic composition obtained by database searching |
|psms| 5 | Final PSMs format output with NovoLign annotation |

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

