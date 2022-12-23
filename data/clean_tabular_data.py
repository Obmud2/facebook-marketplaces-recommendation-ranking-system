import pandas as pd
from torch.utils.data import Dataset

class FBMDatasets(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.categories = self.import_categories()
        self.generate_labels_csv()

    def import_images(self) -> pd.DataFrame:
        df = pd.read_csv("data/raw/Images.csv")
        return df

    def import_products(self) -> pd.DataFrame:
        df = pd.read_csv("data/raw/Products.csv", lineterminator="\n")
        df['price'] = df['price'].str.replace(r'[^0-9\.]+', '', regex=True).astype('float')
        df['category_top'] = df['category'].map(lambda cat: cat.split("/")[0].strip())
        df['category_top'] = df['category_top'].map(lambda cat: self.encode_category(cat))
        return df

    def import_categories(self) -> dict:
        categories = pd.read_csv("data/categories.csv", delimiter=";")
        categories = pd.Series(data=categories.text_category.values, index=categories.coded_category).to_dict()
        return categories

    def encode_category(self, cat_str) -> int:
        for k, v in self.categories.items():
            if v == cat_str:
                return k
        raise ValueError("Category not found")

    def decode_category(self, cat_int) -> str:
        try:
            return self.categories[cat_int]
        except:
            raise ValueError("Category not found")

    def generate_labels_csv(self):
        i = self.import_images()
        p = self.import_products()[['id', 'category_top']]
        p = pd.Series(data=p.category_top.values, index=p.id).to_dict()
        i['category'] = i['product_id'].map(lambda p_id: p[p_id])
        i['id'] += '_resized.jpg'
        i[['id', 'category']].to_csv("data/labels.csv", index=False)

if __name__ == "__main__":
    i = FBMDatasets()



    

