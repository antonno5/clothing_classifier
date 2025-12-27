from test import main as test

import fire

from train import main as train


if __name__ == "__main__":
    fire.Fire({"train": train, "test": test})
