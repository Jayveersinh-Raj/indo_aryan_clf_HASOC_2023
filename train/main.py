import yaml
from load_dataset import load_data
import torch
from pretrain import to_tensordataset, data_loader, init_optimizer
from train import train
from load_model import load_checkpoints
from transformers import get_linear_schedule_with_warmup
import os


def save_model(model, tokenizer) -> None:
  """
  A function to store the model and files on a specified path from paths.yaml

  Parameters:
  -----------
  model: Trained model
  tokenizer: the tokenizer

  Returns:
  -----------
  None
  """

  # Read the hyper-parameters from yaml file using with context manager
  with open('paths.yaml') as parameters:
    path = yaml.load(parameters, Loader=yaml.FullLoader)

  # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

  output_dir = path["output_dir"]
  
  # Create output directory if needed
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  
  print("Saving model to %s" % output_dir)
  
  # Save a trained model, configuration and tokenizer using `save_pretrained()`.
  # They can then be reloaded using `from_pretrained()`
  model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)




def pretraining_steps(device: str):
  """
  Function that implements all the pretraining steps, and calls the trainer.
  Also saves the trained model to a specified directory

  Parameters:
  -----------
  device: cpu or gpu

  Returns:
  -----------
  None
  """

  # Read the hyper-parameters from yaml file using with context manager
  with open('hyperparameters.yaml') as parameters:
    hyper_parameter = yaml.load(parameters, Loader=yaml.FullLoader)
  
  # Load dataset from the path
  dataset = load_data()

  # Load the model and the tokenizer
  model, tokenizer = load_checkpoints()

  # Form the tensor dataset
  dataset = to_tensordataset(dataset=dataset, 
                             labels = dataset["train"][hyper_parameter["label_column"]],
                            tokenizer = tokenizer)

  # Data loader for training                            
  train_dataloader = data_loader(dataset = dataset, 
                                 batch_size = hyper_parameter["batch_size"])

  # Optimizer
  optimizer = init_optimizer(model = model)      
  
  # Epochs and the scheduler
  epochs = hyper_parameter["epochs"]
  total_steps = len(train_dataloader) * epochs
  
  # Create the learning rate scheduler.
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)  

   
  # Training and saving the model
  train(epochs, model, train_dataloader, optimizer, scheduler, device)

  # Save the trained model, and the tokenizer
  save_model(model, tokenizer)


def main():
  # If there's a GPU available...
  if torch.cuda.is_available():
  
      # Tell PyTorch to use the GPU.
      device = torch.device("cuda")
  
      print('There are %d GPU(s) available.' % torch.cuda.device_count())
  
      print('We will use the GPU:', torch.cuda.get_device_name(0))
  
  # If not...
  else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")

  pretraining_steps(device=device)


# Call the main function to run everything
main()