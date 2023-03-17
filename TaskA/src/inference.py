# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023/03/17

# Importing libraries
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from torch import cuda
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


# Setting up the device for GPU/CPU usage
device = 'cuda:7' if cuda.is_available() else 'cpu'
print(f'Current device: {device}')


# Creating a custom dataset for reading the dataframe and loading it into the dataloader 
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, dialogue_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.dialogue_len = dialogue_len
        self.dialogue = self.data.dialogue

    def __len__(self):
        return len(self.dialogue)

    def __getitem__(self, index):
        # Normalization
        source = str(self.dialogue[index])
        source = ' '.join(source.split())

        # Encode
        source = self.tokenizer.batch_encode_plus([source], max_length=self.dialogue_len, padding='max_length', truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        return {
            'source_ids':   source_ids.to(dtype=torch.long), 
            'source_mask':  source_mask.to(dtype=torch.long),
        }

# Inference 
def inference(tokenizer, model, device, loader):
    model.eval()

    predictions = []
    sources = []

    with torch.no_grad():
        for index, data in enumerate(tqdm(loader, desc="Validation Processing"), 0):
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=256, 
                num_beams=3,
                repetition_penalty=2.0, 
                length_penalty=1.0, 
                early_stopping=True
            )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            source = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)for s in ids]

            predictions.extend(preds)
            sources.extend(source)

    return predictions, sources

def main(input_path, output_path):
    # Tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("Elfsong/DocTor5")

    # Defining the model
    model = T5ForConditionalGeneration.from_pretrained("Elfsong/DocTor5")
    
    # Move the model to devices
    # Option 1. Parallelize the model
    # device_map = {
    #     0: [0,  1,  2,  3,  4,  5],
    #     1: [6,  7,  8,  9,  10, 11],
    #     2: [12, 13, 14, 15, 16, 17],
    #     3: [18, 19, 20, 21, 22, 23],
    # }
    # model.parallelize(device_map)

    # # Option 2. Solo
    model.to(device)

    validation_df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
    val_set = CustomDataset(validation_df, tokenizer, 512)

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    val_loader = DataLoader(val_set, **val_params)

    predictions, sources = inference(tokenizer, model, device, val_loader)

    # Postprocessing
    section_name_list, section_text_list = list(), list()
    for p in predictions:
        if p.startswith("CC"):
            section_name, section_text = p.split(": ")[0], ": ".join(p.split(": ")[1:])
            section_name_list += [section_name]
            section_text_list += [section_text]
        else:
            section_name, section_text = p.split(" : ")[0], " : ".join(p.split(" : ")[1:])
            section_name_list += [section_name]
            section_text_list += [section_text]

    final_df = pd.DataFrame(list(zip(validation_df.ID, section_name_list, section_text_list)), columns =['TestID', 'SystemOutput1', 'SystemOutput2'])

    final_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    print("Let's get ready to rumble!")
    print("[-] As for the first time run, it may take 30 mins to download model files.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input file path')
    parser.add_argument('--output', type=str, help='output file path')
    args = parser.parse_args()

    main(args.input, args.output)