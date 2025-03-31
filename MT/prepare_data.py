import pandas as pd
import csv
import os
os.chdir('D:/Prjs/AIT3001/MT/data_origin')

def merge_files(en_file, vi_file, output_file):
    with open(en_file, 'r', encoding='utf-8') as f_en, open(vi_file, 'r', encoding='utf-8') as f_vi:
        en_lines = [line.strip().replace("&quot;", '"').replace(" &apos;", "'") for line in f_en]
        vi_lines = [line.strip().replace("&quot;", '"').replace(" &apos;", "'") for line in f_vi]

    assert len(en_lines) == len(vi_lines), "The number of lines in English and Vietnamese files do not match."

    df = pd.DataFrame({
        'English': en_lines,
        'Vietnamese': vi_lines
    })

    df.to_csv(output_file, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"Saved merged data to {output_file}")

def sample_data(input_file, output_file, num_samples):
    df = pd.read_csv(input_file, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
    df.dropna(how='any', inplace=True)

    sampled_df = df.sample(n=num_samples, random_state=42)

    sampled_df.to_csv(output_file, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"Saved sampled data to {output_file}")

def combine_data(*input_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in input_files:
            df = pd.read_csv(file, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
            for _, row in df.iterrows():
                f.write(row['English'] + '\n')
                f.write(row['Vietnamese'] + '\n')
    print(f"Saved full data to {output_file}")
    
if __name__ == "__main__":
    
    merge_files('train.en', 'train.vi', 'train_merged.tsv')
    merge_files('val.en', 'val.vi', 'val_merged.tsv')
    merge_files('test.en', 'test.vi', 'test_merged.tsv')

    sample_data('train_merged.tsv', 'train.tsv', 7500)
    sample_data('val_merged.tsv', 'val.tsv', 500)
    sample_data('test_merged.tsv', 'test.tsv', 1000)
    
    combine_data('train.tsv', 'val.tsv', 'test.tsv', output_file='dataset.txt')