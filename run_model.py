from utils import run_pipeline

input_path = r'wgan_negative_based_with_stats.csv'
first_output = 'wgan_negative_based_with_stats_emb.csv'
second_output = 'wgan_negative_based_with_stats_classification_results.csv'
sequence_column = 'sequence'

if __name__ == "__main__":
    run_pipeline(
        input_csv_path=input_path,
        first_output=first_output,
        second_output=second_output,
        model_name='zhihan1996/DNABERT-2-117M',
        model_path=r'model/LGBM_model.pkl',
        sequence_column=sequence_column
    )



