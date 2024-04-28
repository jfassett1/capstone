import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from datetime import datetime
import os
import gc
import pathlib
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()



data_dir = pathlib.Path(__file__).parent.parent / "data"
df = pd.read_csv(data_dir/"transformed/data.csv")
df = df.dropna()

# Removing unnecessary columns
cols = set(df.columns)
metadata = set(['followers','retweet','hash','link','capitals','numerics','special_chars','exclamations','avg_word_length','sentences'])
unneeded = cols - metadata
metadata = df.drop(columns=unneeded)
metadata = metadata.to_numpy()
def vectorize_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Pool the outputs and convert to numpy array
    # Here, we simply take the mean of the second to last hidden layer
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings

textcol = df['content'].iloc[:-1000]
# print(len(textcol))

# x = textcol.iloc[0:100]

# p = vectorize_text(list(x),tokenizer,model)

# print(p.shape)
# exit()
batches = 5000
text_length = len(textcol)
base_batch_size = text_length // batches
remaining_items = text_length % batches

pbar = tqdm(range(batches), colour="green", desc="Starting batches...")

for batch in pbar:
    start_idx = batch * base_batch_size + min(batch, remaining_items)
    if batch < remaining_items:
        end_idx = start_idx + base_batch_size + 1
    else:
        end_idx = start_idx + base_batch_size

    batch_text = textcol.iloc[start_idx:end_idx]
    batch_metadata = metadata[start_idx:end_idx]

    embedded_content_array = vectorize_text(list(batch_text), tokenizer, model)
    embedded_content_array = np.hstack([embedded_content_array, batch_metadata])
    size = embedded_content_array.shape[0]
    pbar.set_description(f"Batch {batch}, Size {size}")

    np.save(data_dir / f'embedded/embedded_text{batch}.npy', embedded_content_array)
    torch.cuda.empty_cache()  # Clear GPU cache after each batch to free up memory





exit()
batches = 5000
batch_size = len(textcol) // batches
pbar = tqdm(range(batches), colour="green", desc="Starting batches...")

size = 0
for batch in pbar:
    pbar.set_description(f"Batch {batch}, Size {size}")

    # tqdm.write(f"\rProcessing Batch {batch+1}/{batches}\n")
    start_idx = batch * batch_size
    # Lose a few samples, but doesnt matter ultimately
    end_idx = start_idx + batch_size

    #Defining dataframe and embedded vector. Will combine these and save as pyarrow file
    batch_text = textcol.iloc[start_idx:end_idx]
    batch_metadata = metadata[start_idx:end_idx]
    # print("Current Batch memory use:",dfbatch['content'].memory_usage()/1000,"MB")

    embedded_content_array = vectorize_text(list(batch_text), tokenizer, model)
    embedded_content_array = np.hstack([embedded_content_array,batch_metadata])
    size = embedded_content_array.shape[0]
    # tqdm.write(f"Batch {batch+1} Complete -- Time: {datetime.now()}\n")
    np.save(data_dir/f'embedded/embedded_text{batch}.npy', embedded_content_array)
    torch.cuda.empty_cache()


