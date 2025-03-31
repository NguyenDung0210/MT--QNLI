import pandas as pd
import os

data_dir = r"Final Prj\QNLIv2_origin"

input_file = os.path.join(data_dir, "test.tsv")
output_file = os.path.join(data_dir, "test2.tsv")

df = pd.read_csv(input_file, delimiter='\t', header=0, quoting=3)
df2 = df.drop(columns='index')
df2.to_csv(output_file, sep='\t', index=False)

print(df)
print(df2)
