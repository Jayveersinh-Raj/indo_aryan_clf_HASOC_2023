from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
import numpy as np
import time
import datetime

def to_tensordataset(dataset, labels, tokenizer):
  input_ids = []
  attention_masks = []
  
  # For every sentence...
  for sent in dataset["train"]["Sentences"]:
      # encode_plus will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the [CLS] token to the start.
      #   (3) Append the [SEP] token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to max_length
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 128,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                     )
  
      # Add the encoded sentence to the list.
      input_ids.append(encoded_dict['input_ids'])
  
      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])
  
  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)
  return TensorDataset(input_ids, attention_masks, labels)


def data_loader(dataset, batch_size):
  # The DataLoader needs to know our batch size for training, so we specify it
  # here. For fine-tuning BERT on a specific task, the authors recommend a batch
  # size of 16 or 32.
  
  # Create the DataLoaders for our training and validation sets.
  # We'll take training samples in random order.
  train_dataloader = DataLoader(
              dataset,  # The training samples.
              sampler = RandomSampler(dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )
    
  return train_dataloader

def init_optimizer(model):
  
  # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
  # I believe the 'W' stands for 'Weight Decay fix"
  return AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                  )


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)                  



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
