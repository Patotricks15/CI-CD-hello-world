from model import create_pipeline

if __name__ == "__main__":
    pipeline, X_test, y_test = create_pipeline()
    score = pipeline.score(X_test, y_test)
    print(f"Accuracy: {score:.4f}")
