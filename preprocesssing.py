import os
from git.repo.base import Repo
import pandas as pd

def preprocess():
    if not os.path.exists("short-jokes-dataset"):
        Repo.clone_from("https://github.com/amoudgl/short-jokes-dataset.git", "short-jokes-dataset")

        df = pd.read_csv("short-jokes-dataset/shortjokes.csv")

        with open("jokes.txt", "w+") as f:
            for joke in df["Joke"]:
                f.write(joke + "\n")
        print("Preprocessing done")
    else:
        print("Preprocessing already done")


if __name__ == "__main__":
    preprocess()
