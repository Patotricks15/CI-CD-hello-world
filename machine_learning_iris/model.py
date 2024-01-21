from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def create_pipeline():
    # Carregar o dataset Iris
    data = load_iris()
    X, y = data.data, data.target

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar o pipeline de machine learning
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test
