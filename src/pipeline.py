import pandas as pd
from sklearn.model_selection import train_test_split
from src.eda import perform_eda
from src.preprocessing import preprocess_features
from src.tokenizer import tokenize, build_vocab
from src.stemming import apply_stemming
from src.llm_model import generate_embeddings
from src.loan_model import train_model, evaluate_model, save_model
from src.fairness_check import check_fairness


def run_pipeline(csv_path: str):
    print("\n" + "=" * 50)
    print(" 🚀 INITIALIZING LOAN PREDICTION PIPELINE")
    print("=" * 50)

    print("[1/7] Loading Data...")
    df = pd.read_csv(csv_path)

    print("[2/7] Running EDA (Outputs saved to /eda_outputs)...")
    perform_eda(df, target_col="loan_amount")

    print("[3/7] Preprocessing & Engineering Financial Features...")
    df_clean = preprocess_features(df)

    print("[4/7] Compiling Custom Transformer Text Embeddings...")
    raw_texts = df_clean['loan_notes'].tolist()
    tokenized_texts = [tokenize(txt) for txt in raw_texts]
    stemmed_texts = [apply_stemming(tokens) for tokens in tokenized_texts]

    vocab = build_vocab(stemmed_texts)
    embeddings = generate_embeddings(stemmed_texts, vocab)

    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    X = pd.concat([df_clean.drop(columns=['loan_id', 'loan_amount', 'loan_notes']), emb_df], axis=1)
    y = df_clean['loan_amount']

    # Store reference demographic data for fairness checks
    demo_cols = [c for c in X.columns if 'education' in c or 'self_employed' in c]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[5/7] Training Gradient Boosting Regressor...")
    model = train_model(X_train, y_train)
    save_model(model)

    print("[6/7] Evaluating Model Performance...")
    preds, mae, r2 = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 50)
    print(" 📊 MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"  Accuracy (R² Score):    {r2 * 100:.2f}%")
    print(f"  Mean Absolute Error:    ${mae:,.2f}")

    print("\n[7/7] Executing Bias & Fairness Diagnostics...")
    df_eval = X_test.copy()
    df_eval['loan_amount'] = y_test
    df_eval['predictions'] = preds

    check_fairness(df_eval, target_col='loan_amount', pred_col='predictions', group_cols=demo_cols)
    print("\n✅ PIPELINE EXECUTION COMPLETE.\n")