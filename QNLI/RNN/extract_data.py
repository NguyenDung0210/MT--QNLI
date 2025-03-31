import pandas as pd
import os

def reduce_dataset(input_file, output_file, sample_size):
    data = pd.read_csv(input_file, delimiter='\t', header=0, quoting=3)
    data = data.drop(columns=['index'])
    sampled_data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
    sampled_data.to_csv(output_file, sep='\t', index=False)

def split_dev_dataset(input_file, val_output_file, test_output_file):
    data = pd.read_csv(input_file, delimiter='\t', header=0, quoting=3)
    data = data.drop(columns=['index']) 

    val_data = data.sample(n=1000, random_state=42).reset_index(drop=True)
    remaining_data = data.drop(val_data.index).reset_index(drop=True) 
    test_data = remaining_data.sample(n=1000, random_state=24).reset_index(drop=True)

    val_data.to_csv(val_output_file, sep='\t', index=False)
    test_data.to_csv(test_output_file, sep='\t', index=False)

def main():
    data_dir = r"Final Prj\QNLIv2_origin"
    output_dir = r"Final Prj\QNLI"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo tập train với 8000 mẫu
    train_input_file = os.path.join(data_dir, "train.tsv")
    train_output_file = os.path.join(output_dir, "train.tsv")
    reduce_dataset(train_input_file, train_output_file, sample_size=8000)

    dev_input_file = os.path.join(data_dir, "dev.tsv")
    val_output_file = os.path.join(output_dir, "val.tsv")
    test_output_file = os.path.join(output_dir, "test.tsv")
    
    # Chia tập dev thành val và test mỗi tập 1000 mẫu
    split_dev_dataset(dev_input_file, val_output_file, test_output_file)
    

if __name__ == "__main__":
    main()