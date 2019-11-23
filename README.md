# mat-rbf

This is an unoptimized toy implementation of our SIGGRAPH ASIA 2019
paper, **Material-adapted Refinable Basis Functions for Elasticity
Simulation**, for 2D elasticity with regular grid
discretization only. After compilation, one could run the program by
```
./build/main config.json
```
The results will be written to `vtk`
formatted files, which can be visualized by `paraview`.

To cite our paper, you could use the following BibTex entry.
```
@article{Chen:2019:MRB:3355089.3356567,
 author = {Chen, Jiong and Budninskiy, Max and Owhadi, Houman and Bao, Hujun and Huang, Jin and Desbrun, Mathieu},
 title = {Material-adapted Refinable Basis Functions for Elasticity Simulation},
 journal = {ACM Trans. Graph.},
 issue_date = {November 2019},
 volume = {38},
 number = {6},
 month = nov,
 year = {2019},
 issn = {0730-0301},
 pages = {161:1--161:15},
 articleno = {161},
 numpages = {15},
 url = {http://doi.acm.org/10.1145/3355089.3356567},
 doi = {10.1145/3355089.3356567},
 acmid = {3356567},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {deformable body simulation, material-adapted basis functions, numerical coarsening, operator-adapted wavelets},
}
```

Enjoy yourself for hacking/optimizing the code:)