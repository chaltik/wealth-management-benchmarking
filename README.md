# wealth-management-benchmarking

Supporting material for the talk at USF 2023 DSCO conference on March 12-14, 2023.
[Slides](https://docs.google.com/presentation/d/12Crkv4Z4dEtlaUO5aMwsxx3SPhJS0S4OpQ0EU_Hpc0Y/edit?usp=share_link)

## Running the code ##
1. create virtual environment with `requirements.txt` 
  (e.g. using conda:
  `conda create -n myenv python=3.9 pip`
  
  `conda activate myenv`
  
  `pip install -r requirements.txt`
  
  `python -m ipykernel install --name myenv --user`
  
  `conda deactivate`
2. Create your own file `api_keys.json` using `api_keys_example.json` as an example. Currently the only key needed is for FRED API which can be obtained (free) at [FRED website](https://fredaccount.stlouisfed.org/apikeys).
3. Launch jupyter notebook or jupyter lab and pick `myenv` from the list of kernel.

