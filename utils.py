import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import json
import os
import tqdm
import joblib
import lightgbm as lgb
from constants import selected_features


def load_data(csv_path: str, sequence_column: str):
    df = pd.read_csv(csv_path)
    sequences = df[sequence_column].dropna().unique()
    return sequences


def initialize_model(model_name: str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, tokenizer, device


def embed_sequences(sequences, model, tokenizer, device):
    results = []

    with torch.no_grad():
        for sequence in tqdm.tqdm(sequences, desc="Embedding sequences"):
            tokens = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True).to(device)

            if tokens['input_ids'].shape[1] > tokenizer.model_max_length:
                print(f"Skipping long sequence of length {len(sequence)}")
                continue

            outputs = model(**tokens)
            hidden_states = outputs[0].squeeze(0)
            emb = torch.mean(hidden_states, dim=0)
            emb = emb / emb.norm()  

            results.append({
                "sequence": sequence,
                "embedding": emb.tolist()
            })

    return results


def build_final_dataframe(results: list):
    df = pd.DataFrame(results)
    emb_df = pd.DataFrame(df["embedding"].tolist())
    emb_df.columns = [f"emb_{i}" for i in emb_df.columns]

    final_df = pd.concat([df["sequence"], emb_df], axis=1)
    return final_df

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at path: {model_path}")

    model = joblib.load(model_path)
    return model

def predict_on_embeddings(
    df,
    model_path: str,
    output_path: str
    ):
   
    model = load_model(model_path)

    # Предсказания
    X = df[selected_features]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    df["y_pred"] = y_pred
    df["y_proba"] = y_proba

    result_df = df[["sequence", "y_pred", "y_proba"]]
    result_df = result_df[result_df["y_proba"] > 0.95]


    result_df.to_csv(output_path, index=False)
    print(f"Results saved to: {os.path.abspath(output_path)}")

    return result_df

def run_pipeline(
    input_csv_path: str,
    first_output: str,
    second_output: str,
    model_name: str,
    model_path: str,
    sequence_column: str
):
    sequences = load_data(input_csv_path, sequence_column)

    model, tokenizer, device = initialize_model(model_name)

    results = embed_sequences(sequences, model, tokenizer, device)

    full_df = build_final_dataframe(results)
    full_df.to_csv(first_output)
    prediction_df = predict_on_embeddings(full_df, model_path, second_output)
    

