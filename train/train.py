import time
import datetime
import random
import numpy as np
import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import random
import numpy as np

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

def train(epochs, model, train_dataloader, optimizer, scheduler, device):
  # Set the seed value all over the place to make this reproducible.
  seed_val = 42
  
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  
  # We'll store a number of quantities such as training and validation loss,
  # validation accuracy, and timings.
  training_stats = []
  
  # Measure the total training time for the whole run.
  total_t0 = time.time()
  
  # For each epoch...
  for epoch_i in range(0, epochs):
  
      # ========================================
      #               Training
      # ========================================
  
      # Perform one full pass over the training set.
  
      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')
  
      # Measure how long the training epoch takes.
      t0 = time.time()
  
      # Reset the total loss for this epoch.
      total_train_loss = 0
  
      # Put the model into training mode. Don't be mislead--the call to
      # `train` just changes the *mode*, it doesn't *perform* the training.
      # `dropout` and `batchnorm` layers behave differently during training
      # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
      model.train()
  
      # For each batch of training data...
      for step, batch in enumerate(train_dataloader):
  
          # Progress update every 40 batches.
          if step % 40 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
  
              # Report progress.
              print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
  
          # Unpack this training batch from our dataloader.
          #
          # As we unpack the batch, we'll also copy each tensor to the GPU using the
          # `to` method.
          #
          # `batch` contains three pytorch tensors:
          #   [0]: input ids
          #   [1]: attention masks
          #   [2]: labels
          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)
  
          # Always clear any previously calculated gradients before performing a
          # backward pass. PyTorch doesn't do this automatically because
          # accumulating the gradients is "convenient while training RNNs".
          # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
          model.zero_grad()
  
          # Perform a forward pass (evaluate the model on this training batch).
          # The documentation for this `model` function is here:
          # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
          # It returns different numbers of parameters depending on what arguments
          # arge given and what flags are set. For our useage here, it returns
          # the loss (because we provided labels) and the "logits"--the model
          # outputs prior to activation.
          outputs = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)
  
          loss = outputs[0]
          total_train_loss += loss.item()
  
          # Perform a backward pass to calculate the gradients.
          loss.backward()
  
          # Clip the norm of the gradients to 1.0.
          # This is to help prevent the "exploding gradients" problem.
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  
          # Update parameters and take a step using the computed gradient.
          # The optimizer dictates the "update rule"--how the parameters are
          # modified based on their gradients, the learning rate, etc.
          optimizer.step()
  
          # Update the learning rate.
          scheduler.step()
  
      # Calculate the average loss over all of the batches.
      avg_train_loss = total_train_loss / len(train_dataloader)
  
      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)