import os
import pandas as pd

os.system("!git clone https://github.com/amoudgl/short-jokes-dataset.git")

df = pd.read_csv("short-jokes-dataset/shortjokes.csv")

with open("jokes.txt", "w+") as f:
    for joke in df["Joke"]:
        f.write(joke + "\n")
print("Done")
