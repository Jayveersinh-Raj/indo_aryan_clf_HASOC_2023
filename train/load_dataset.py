import yaml
from datasets import load_dataset

with open('paths.yaml') as parameters:
  path = yaml.load(parameters, Loader=yaml.FullLoader)

def load_data(csv = True):
  if csv == True:
    return load_dataset("csv", data_files=path["dataset_path"])

  else:
    return load_dataset(data_files=path["dataset_path"])