"""Строит DNABERT-2 эмбеддинги для очищенного CSV-файла."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "data" / "sequence_craft_clean.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "sequence_craft_clean_emb.csv",
    )
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--model", default=MODEL_NAME)
    return parser.parse_args()


def load_dnabert(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    return model, tokenizer, device


def embed_sequences(sequences, model, tokenizer, device):
    embeddings = {}

    with torch.inference_mode():
        for sequence in tqdm(sequences, desc="Embedding sequences"):
            tokens = tokenizer(sequence, return_tensors="pt").to(device)
            if tokens["input_ids"].shape[1] > tokenizer.model_max_length:
                raise ValueError(
                    f"Sequence length {len(sequence)} exceeds the model limit "
                    f"({tokenizer.model_max_length} tokens)."
                )

            hidden_states = model(**tokens)[0].squeeze(0)
            embedding = hidden_states.mean(dim=0)
            embedding = embedding / embedding.norm()
            embeddings[sequence] = embedding.cpu().tolist()

    return embeddings


def main():
    args = parse_args()
    data = pd.read_csv(args.input)

    if args.sequence_column not in data.columns:
        raise ValueError(f"Column '{args.sequence_column}' is missing from {args.input}")

    sequences = data[args.sequence_column]
    if sequences.isna().any():
        raise ValueError(f"Column '{args.sequence_column}' contains missing values")

    sequences = sequences.astype(str)
    model, tokenizer, device = load_dnabert(args.model)
    embeddings = embed_sequences(sequences.unique(), model, tokenizer, device)
    embedding_df = pd.DataFrame(
        sequences.map(embeddings).tolist(),
        columns=[f"emb_{i}" for i in range(model.config.hidden_size)],
    )
    result = pd.concat([data.reset_index(drop=True), embedding_df], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
