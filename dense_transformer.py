class DenseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X
