import pandas as pd

main_dataset = pd.read_csv('data/dataset_csv_file/dataset.csv')

file_path_df = main_dataset['filepath']
damage_location_df = main_dataset['damage_location']
damage_severity_df = main_dataset['damage_severity']
caption_df = main_dataset['caption']

df_dict = {
    'filepath': file_path_df,
    "damage_location": damage_location_df,
    'damage_severity': damage_severity_df
}

df_labeling = pd.DataFrame(df_dict, columns=['filepath', 'damage_location', 'damage_severity'])
df_labeling.to_csv("data/dataset_csv_file/dataset_labeling.csv", index=False)

df_dict = {
    "filepath": file_path_df
}

df_filepath = pd.DataFrame(df_dict, columns=['filepath'])
df_filepath.to_csv("data/dataset_csv_file/dataset_filepath.csv", index=False)

df_dict = {
    'filepath': file_path_df,
    'caption': caption_df
}

df_captioning = pd.DataFrame(df_dict, columns=['filepath', 'caption'])
df_captioning.to_csv("data/dataset_csv_file/dataset_captioning.csv", index=False)