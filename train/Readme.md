# This is the folder that contains training files.
# How to use it (run the main.py)

    python3 main.py
   
# Description of the yamls
- `paths.yaml`: contains paths for checkpoints, dataset, saving model directory path, etc.
- `hyperparameters.yaml`: contains the hyperparameters like epochs, batch_size, column name containing labels

# The flow of the files:
- `pretrain.py`: Implements the necessary functions before training, like converting to a tensor dataset, creating a data loader, etc.
- `train.py`: Implements the main training loop
- `load_dataset.py`: Implements the loading of the dataset either from a csv on a local machine or from Huggingface Hub.
- `load_model.py`: Implements the load checkpoints code to load the model checkpoints and the tokenizer.
- `main.py`: The driver file and code to put the pipeline together and save the trained model files to the specified path.
