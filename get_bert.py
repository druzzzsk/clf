import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import tqdm
import json
import os
import warnings 
warnings.filterwarnings('ignore')

final_csv_path = "active_dnazymes_emb.csv"
df = pd.read_csv('files/active_DNAzymes.csv')
unique_sequences = df['sequence'].unique()
#labels = df['target']

# 2. Подключение к CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Загрузка модели и токенизатора
with torch.no_grad():
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model.eval()

    results = []

    for sequence in tqdm.tqdm(unique_sequences):
        tokens = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True).to(device)

        if tokens['input_ids'].shape[1] > tokenizer.model_max_length:
            print(f"Skipping long sequence of length {len(sequence)}")
            continue

        outputs = model(**tokens)
        hidden_states = outputs[0].squeeze(0)

        emb = torch.mean(hidden_states, dim=0)
        emb = emb / emb.norm()  # Нормализация (опционально)

        results.append({
            "input": sequence,
            "embedding": emb.tolist()
        })

# 4. Сохранение в JSON
json_path = "gene_hyenadna.json"
with open(json_path, 'w') as f:
    json.dump(results, f)

# 5. Загрузка из JSON и преобразование в DataFrame
with open(json_path, "r") as f:
    data = json.load(f)

df = pd.json_normalize(data)  # Получим колонки 'input' и 'embedding'

# 6. Разделение embedding на отдельные колонки
embedding_df = pd.DataFrame(df["embedding"].tolist())
embedding_df.columns = [f"emb_{i}" for i in range(embedding_df.shape[1])]

# 7. Объединение с исходной последовательностью
final_df = pd.concat([df["input"], embedding_df], axis=1)
final_df = final_df.rename(columns={"input": "sequence"})
#final_df['target'] = labels
# 8. Сохранение в CSV
final_df.to_csv(final_csv_path, index=False)

print(f"Файл сохранён: {os.path.abspath(final_csv_path)}")