import yaml
from src.video_pipeline import run_pipeline

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(config)
