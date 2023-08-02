import yaml
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

with open('paths.yaml') as parameters:
  path = yaml.load(parameters, Loader=yaml.FullLoader)

def load_checkpoints(checkpoint : str = path["checkpoints_path"]) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
  """
  Loads and return the model, and the tokenizer

  Parameters:
  -----------
  None

  Returns:
  model: An encoder only model stacked with a classifier preferably pretrained
  tokenizer: The tokenizer corresponding to the model
  """

  checkpoint = checkpoint
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  model.cuda()
  return model, tokenizer