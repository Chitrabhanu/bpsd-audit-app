import os
import yaml

def load_config():
    env = os.getenv("APP_ENV", "local")
    path = f"configs/{env}.yaml"

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["env"] = env
    return config