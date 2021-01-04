# procgenac
Project for testing out Actor-Critic methods on Procgen game environments.

## Recreating the results

Clone the repository to the desired destination

`$ git clone https://github.com/jonasjuhlern/procgenac.git`

If you do not want to install it in your base python environment then activate another

(optional) `$ source /path/to/your_python_env/bin/activate`

Then pip install the project itself either in an environment or in your base env (from the root folder).

`$ cd /path/to/procgenac`

`$ pip install .`

Ensure that the project has been correctly installed by running the following from python

    import procgenac
    print(procgenac.__package__)

This should print 'procgenac' 

### Recreating the results from Jupyter notebook

Simply run the notebook located at `scripts/recreate_results.ipynb`. Be sure to change the kernel to one where the project is installed.

### Recreating the results for a specific hyperparameter configuration

Run the file `procgenac/hpcfiles/train.py` with your desired hyperparameters replaced with `<hyperparameter>`

    python train.py --model_type <model_type> --env_name <env_name> --num_envs <num_envs> \
        --num_levels <num_levels> --feature_dim <feature_dim> --value_coef <value_coef> \
        --entropy_coef <entropy_coef> --eps <eps> --grad_eps <grad_eps> --num_epochs <num_epochs> \
        --batch_size <batch_size> --adam_lr <adam_lr> --adam_eps <adam_eps> --num_steps <num_steps> \
        --total_steps <total_steps> --cnn_type <cnn_type> --model_id <model_id>

When the model has trained the results will be available in folders.

### Recreating the results for all the final moodels from DTU HPC
If you are on DTU HPC the results from the final models can now be recreated by running these commands.

Create the jobscripts that train the 3 models

`$ cd hpcfiles/gridsearch/`

`$ python get_gridsearch_jobscripts.py`

Submit the jobscripts

`$ ./submit_gridsearch_jobs.sh`

The results will now be written to files in 'procgenac/results/'

## Setting up development environment

There are two main ways to set up the environment. Either using conda or not using conda. 

### 1. Using conda

Make sure that conda is up to date with

`$ conda update conda`

Clone the repository to the desired destination

`$ git clone https://github.com/jonasjuhlern/procgenac.git`

Create a development environment directly from the dev_env YAML file.

`$ conda env create --file dev_env.yml`

### 2. Not using conda

If only using pip for package management and using some other tool for environments then first create the environemnt.

`$ some command to create env`

Then pip install the project itself as a development environment (from the root folder).

`$ cd /path/to/procgenac`

`$ pip install .[dev]`
