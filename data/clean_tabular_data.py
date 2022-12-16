import pandas as pd

def import_images() -> pd.DataFrame:
    df = pd.read_csv("data/raw/Images.csv")
    return df

def import_products() -> pd.DataFrame:
    df = pd.read_csv("data/raw/Products.csv", lineterminator="\n")
    df['price'] = df['price'].str.replace(r'[^0-9\.]+', '', regex=True).astype('float')
    return df

if __name__ == "__main__":
    i = import_images()
    p = import_products()
    