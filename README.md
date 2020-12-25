# procgenac
Project for testing out Actor-Critic methods on Procgen game environments.


## How to et up development environment

There are two main ways to set up the environment. Either using conda or not using conda. 

### 1. Using conda

Make sure that conda is up to date with

`$ conda update conda`

Clone the repository to the desired destination

`git clone https://github.com/jonasjuhlern/procgenac.git`

Create a development environment directly from the dev_env YAML file.

`conda env create --file dev_env.yml`

### 2. Not using conda

If only using pip for package management and using some other tool for environments then first create the environemnt.

`some command to create env`

Then pip install the project itself as a development environment (from the root folder).

`cd /path/to/procgenac`

`pip install .[dev]`
