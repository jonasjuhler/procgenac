import pandas as pd
from itertools import product
from collections import defaultdict

model_id = 0


class GridsearcHyperparameters:
    def __init__(self, model_type):
        grad_eps = {"PPO": ["0.7"], "A2C": ["0.5"]}
        adam_lrs = {"PPO": ["5e-5"], "A2C": ["5e-5"]}
        adam_eps = {"PPO": ["1e-5"], "A2C": ["1e-3"]}
        cnn_type = {"PPO": ["impala", "nature"], "A2C": ["impala"]}
        num_levels = {"PPO": ["200"], "A2C": ["200"]}

        self.paramater_dict = {
            "model_type": [model_type],
            "env_name": ["starpilot"],
            "num_envs": ["32"],
            "num_levels": num_levels[model_type],
            "feature_dim": ["128"],
            "value_coef": ["0.5"],
            "entropy_coef": ["0.01"],
            "eps": ["0.2"],
            "grad_eps": grad_eps[model_type],
            "adam_lr": adam_lrs[model_type],
            "adam_eps": adam_eps[model_type],
            "num_epochs": ["3"],
            "batch_size": ["512"],
            "num_steps": ["256"],
            "total_steps": ["25_000_000"],
            "cnn_type": cnn_type[model_type],
        }
        self.columns = list(self.paramater_dict.keys())

    def iterate(self):
        keys, values = zip(*self.paramater_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
        for param_dict in permutations_dicts:
            yield param_dict


param_configs = []
mts = ["A2C", "PPO"]
models_dict = defaultdict(list)

for mt in mts:
    pars = GridsearcHyperparameters(mt)

    template = open("template.sh", "r").read()

    for par_dict in pars.iterate():
        script = template.replace("<model_id>", str(model_id))
        models_dict["model_id"].append(model_id)
        for k, v in par_dict.items():
            script = script.replace("<" + k + ">", v)
            models_dict[k].append(v)
        with open(f"jobscripts/jobscript_model_{model_id}.sh", "w") as f:
            f.write(script)
        model_id += 1

df = pd.DataFrame.from_dict(models_dict)
df.to_csv("../../results/model_configs.csv", index=False)
