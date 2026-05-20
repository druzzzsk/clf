import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split

# Загрузка исходного датасета с ДНКзимами
def load_and_prepare_data(filepath):

    data = pd.read_csv(filepath)
    data = data[['e', 'kobs']]
    data = data.drop_duplicates(subset='e')
    data = data.dropna()
    
    data.rename(columns={
        'e': 'sequence',
        'kobs': 'target'
    }, inplace=True)
    
    data['target'] = (data['target'] > 1e-07).astype(int)
    data = data[data['target'] == 1] # Оставляем только активные последовательности
    
    return data

# Генерация отрицательного класса
def generate_negative_sequences(data, # Данные с активными последовательностями 
                                seq_col, 
                                num_of_seq, # Количество последовательностей для генерации
                                random_state):
    random.seed(random_state)
    lengths = [len(i) for i in data[seq_col]]

    def generate_random_sequences(length):
        return ''.join(random.choices(['A', 'T', 'C', 'G'], k=length))
    
    sampled_lengths = random.choices(lengths, k=num_of_seq)
    
    # Распределение длинны последовательностей такое же как в исходных данных
    new_sequences = [generate_random_sequences(length) for length in sampled_lengths] 
    generated = pd.DataFrame({'sequence' : new_sequences, 'target':0})
    
    return generated

# Объединение таблиц
def get_final_df(data, generated):
    final_df = pd.concat([data, generated], ignore_index=True)
    return final_df

# Получение дескрипторов для последовательностей
def prepare_seq_features(df: pd.DataFrame, seq_column: str):
    descriptors = get_full_descriptors_for_classifier(
        df, seq_column_name=seq_column
    )
    descriptors.insert(0, "sequence", df[seq_column])
    return descriptors

def make_train_test_split(data):
    X = data.drop('target', axis = 1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

# Итоговый пайплайн
def prepare_final_dataset(filepath: str,
                          num_negatives: int,
                          output_path: str):
  
    data = load_and_prepare_data(filepath)
    generated = generate_negative_sequences(data, seq_col='sequence', num_of_seq=num_negatives, random_state=42)
    final_df = get_final_df(data, generated)

    labels = final_df['target'].copy()

    final_df = prepare_seq_features(final_df, 'sequence')

    final_df['target'] = labels

    X_train, X_test, y_train, y_test = make_train_test_split(final_df)
  
    X_train.to_csv('CAE_X_train.csv')
    X_test.to_csv('CAE_X_test.csv')
    y_train.to_csv('CAE_y_train.csv')
    y_test.to_csv('CAE_y_test.csv')
    final_df.to_csv(output_path, index=False)
    

    return final_df

# Получение данных
df = prepare_final_dataset('data\db_dnazymes_v2.csv', 1000, 'training_data.csv')
