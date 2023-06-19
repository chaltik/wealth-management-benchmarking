# wealth-management-benchmarking

Supporting material for the talk at USF 2023 DSCO conference on March 12-14, 2023.
([Slides](https://www.slideshare.net/KarnChaltikian/goalbasedwealthmanagementbenchmarks20230312pdf))

## Running the code ##
1. create virtual environment with `requirements.txt` 
  (e.g. using conda:
  `conda create -n myenv python=3.9 pip`
  
  `conda activate myenv`
  
  `pip install -r requirements.txt`
  
  `python -m ipykernel install --name myenv --user`
  
  `conda deactivate`
  
2. Create your own file `api_keys.json` using `api_keys_example.json` as an example. Currently the only key needed is for FRED API which can be obtained (free) at [FRED website](https://fredaccount.stlouisfed.org/apikeys).
3. Launch jupyter notebook or jupyter lab and pick `myenv` from the list of kernel. Start tinkering and let me know how it goes.

## Interesting questions / further possible analysis ##

1. For all the three considered profiles using 100% stocks was the winning strategy despite the apparently large differences between their time  horizons,  distribution of goals' over time, and starting conditions. This strongly suggests that in the absence of concern for financial risk "100% stocks" is almost always the best  or nearly the best solution if utility or probability of survival is the maximization objective. Is there a modification of the objective function _not based_ on financial risk that would result in optimal asset allocation trajectory becoming more glidepath-like and allocating to bonds and cash at times? Of course there is always a trivial answer to that question for the  profiles with utility _above_ target: reduce risk until the target is approached from above. The real question is whether anything other than stocks can do better when utility is still below the target. 
2. At the end of the talk (and notebooks) we skectched a proposal for a more complete execution strategy (based on estimating at each decision time the cost of all possible _combinations_ of the remaining goals as opposed to all the goals). This would require introducting the "memory" in the calculation and keeping track of which goals have been commited to so far. 
3. Additional flexibility can be introduced by allowing goals a) to be reduced on the fly within certain limits if needed - as done in other works b) be deferred within certain window. Both are commonly used tactical ools at many a wealth advisor's disposal, and incorporating them into the strategy seems prudent.   
          
## Notable other research on the subject ##
[Sanjiv Das](https://srdas.github.io/) is a renowned authority on the subject of dynamic optimization applications in wealth management - there are several very insightful papers (including one that served as a basis for his talk at the above conference) on his website on this subject.

