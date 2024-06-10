import implicit
import pickle


with open("sparse.pkl", "rb") as sparse:
    sparse_matrix = pickle.load(sparse)

# обучаем
model = implicit.als.AlternatingLeastSquares(
    factors=128, regularization=0.001, iterations=40, calculate_training_loss=False
)
model.fit(sparse_matrix)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
