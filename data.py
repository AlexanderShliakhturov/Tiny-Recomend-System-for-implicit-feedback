from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import pickle


data = pd.read_csv(r"/home/alexander/MAI/wb/data.gzip.csv", compression="gzip")

model_data = data.copy()
model_data.drop(columns=["dt"], inplace=True)
model_data["rating"] = 1

# data_sample = model_data.sample(n=20000000, random_state=42)
data_sample = model_data

# кодируем вручную
person_u = list(np.sort(data_sample.uid.unique()))
thing_u = list(np.sort(data_sample.item_id.unique()))
values = data_sample["rating"].tolist()

row = data_sample.uid.astype("category").cat.set_categories(person_u).cat.codes
col = data_sample.item_id.astype("category").cat.set_categories(thing_u).cat.codes
sparse_matrix = csr_matrix((values, (row, col)), shape=(len(person_u), len(thing_u)))

with open("person.pkl", "wb") as person:
    pickle.dump(person_u, person)

with open("items.pkl", "wb") as items:
    pickle.dump(thing_u, items)

with open("sparse.pkl", "wb") as sparse:
    pickle.dump(sparse_matrix, sparse)
