from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def cross_validate(model_class, data, scaling=False, bins=5, **params):
    best_model = None
    best_score = 0

    for i in range(bins):
        model = None

        if scaling:
            model = make_pipeline(
                StandardScaler(),
                model_class(**params)
            )
        else:
            model = model_class(**params)

        X_train, y_train, X_test, y_test = data.get_data(bins=bins, test_bin=(i + 1))

        model.fit(X_train, y_train)

        score = test_model(model, X_test, y_test)
        print(score)

        if score > best_score:
            best_model = model
            best_score = score

    print("Best:", best_score)
    return best_model

def test_model(model, X_test, y_test):
    preds = model.predict(X_test)

    score = 0
    for i in range(len(preds)):
        if preds[i] == y_test[i]:
            score += 1

    return score/len(preds)