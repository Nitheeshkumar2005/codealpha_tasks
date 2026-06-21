import pickle

model = pickle.load(
    open("models/credit_score_model.pkl", "rb")
)

print(type(model))