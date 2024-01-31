import pandas as pd


def main():
    df = pd.read_csv("./small_imdb.csv")
    print(df.head(5))

if __name__ == '__main__':
    main()