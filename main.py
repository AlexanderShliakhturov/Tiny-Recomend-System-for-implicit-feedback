import pickle
from sys import argv


def main():
    if len(argv) < 3:
        print("Usage: python3 main.py <id пользователя> <количество рекомендаций>")
        return

    try:
        uid_for_rec = int(argv[1])
        n_rec = int(argv[2])
    except ValueError:
        print("id и количество рекомендаций должны быть целочисленными")
        return

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("sparse.pkl", "rb") as sparse:
        sparse_matrix = pickle.load(sparse)

    with open("person.pkl", "rb") as person:
        person_u = pickle.load(person)

    with open("items.pkl", "rb") as item:
        thing_u = pickle.load(item)

    try:
        user_index = person_u.index(uid_for_rec)
    except ValueError:
        print("Несуществующий id пользователя")
        return

    recomendation = model.recommend(
        user_index,
        sparse_matrix,
        N=n_rec,
        filter_already_liked_items=False,
    )

    rec_array = []
    similarity_array = []
    for i in range(n_rec):
        rec_array.append(thing_u[recomendation[0][i]])
        similarity_array.append(recomendation[1][i])

    print(rec_array)


if __name__ == "__main__":
    main()
