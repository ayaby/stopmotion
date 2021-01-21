# Retroactive effects of surprise - code

This repository contains the code required to reproduce results of a registered report testing retroactive effects of surprise. The stimuli and data can be found on OSF: https://osf.io/rjav4/

The repo includes the following files:

1. stopmotion_env.yml: a yaml file that can be used to recreate the conda environment in which the code was run. To use it, download the file and run:  
```conda env create -f /<path_to_yml_file>/stopmotion_env.yml --prefix /<path_for_conda_env>/stopmotion_env```  
Then activate the environment with the following command:  
```conda activate /<path_for_conda_env>/stopmotion_env```  
The environment was created with the help of Johan Carlin (https://github.com/jooh)

2. retro_surprise_analysis.ipynb: a jupyter notebook for reproducing the results (should be opened from within the conda environment). 

3. stopmotion_funcs.py: functions called by the jupyter notebook (the notebook assumes it has been downloaded, and the relevant parameter in the notebook needs to be set to the location of the file)

4. create_brms_models.R: an R script called by the jupyter notebook (the notebook assumes it has been downloaded, and the relevant parameter in the notebook needs to be set to the location of the file). This code is a slight modification of code written by Alex Quent (https://github.com/JAQuent/bayesianMLM/tree/master/CBU_clusterGuide)

* Note that the code requires slurm parallelisation. If anyone wishes to run the code and doesn't have slurm, please contact me.

