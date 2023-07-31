import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import time

def binary_infer(df: pd.DataFrame, column: str, model, tokenizer) -> None:
  """
  A function to take inference for binary classification on a sentence.

  Parameters:
  ------------
  df : The dataframe (test maybe)
  colum : The target column that contains the sentences
  model : The model to get inference on
  tokenizer: The tokenizer to tokenize sentences

  Returns:
  -----------
  None
  """
  
  # Define the device (GPU or CPU)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  # Move the model to the appropriate device
  model = model.to(device)
  
  # Initialize an empty list to store the predictions
  predictions = []
  
  # Define the batch size and total time variables
  batch_size = 100
  total_time = 0.0
  
  # Split the dataframe into batches
  for i in range(0, len(df), batch_size):
      batch_texts = df[column].iloc[i:i+batch_size].tolist()
  
      # Tokenize the batch of texts and move to the device
      inputs = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", max_length=512, truncation=True, padding=True)
      inputs = {key: val.to(device) for key, val in inputs.items()}
  
      # Perform the model inference on the batch
      with torch.no_grad():  # Disable gradient calculation during inference
          start_time = time.time()  # Start time for the batch
          outputs = model(**inputs)
          probabilities = torch.softmax(outputs.logits, dim=1)
          predicted_classes = torch.argmax(probabilities, dim=1)
          end_time = time.time()  # End time for the batch
  
      # Calculate the time taken for the batch
      batch_time = end_time - start_time
  
      # Map predicted class indices to labels
      batch_pred_labels = [1 if pred == 1 else 0 for pred in predicted_classes.tolist()]
  
      # Extend the predictions list with batch results
      predictions.extend(batch_pred_labels)
  
      # Accumulate batch time to the total time
      total_time += batch_time
  
      # Clear variables and tensors that are not required
      del inputs, outputs, probabilities, predicted_classes, batch_texts, batch_pred_labels
      torch.cuda.empty_cache()  # Clear GPU cache if applicable
  
      # Print time taken for the current batch
      print(f"Batch {i//batch_size+1}: {batch_time:.4f} seconds")
  
  # Assign the predictions to the dataframe
  df['pred_lab'] = predictions
  
  # Print total time taken for processing the entire data
  print(f"Total time taken for the entire data: {total_time:.4f} seconds")




# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")



def top_1_accuracy(y_true, y_pred):
    """
    Computes the top-1 accuracy of a classifier.

    Parameters:
    y_true (array-like): True labels of the data.
    y_pred (array-like): Predicted labels of the data.

    Returns:
    top_1_acc (float): The top-1 accuracy of the classifier.
    """
    # Ensure that the inputs have the same shape
    assert y_true.shape == y_pred.shape

    # Calculate the number of correct predictions
    num_correct = (y_true == y_pred).sum()

    # Calculate the top-1 accuracy
    top_1_acc = num_correct / y_true.shape[0]

    return top_1_acc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve



def calculate_and_plot_auc_roc(ground_truth, predicted_probs):
    """
    Calculate the AUC-ROC and plot the ROC curve for a binary classification problem.

    Parameters:
    ground_truth (list): The ground truth labels.
    predicted_probs (list): The predicted probabilities.

    Returns:
    float: The AUC-ROC value.
    """
    # Calculate AUC-ROC
    auc_roc = roc_auc_score(ground_truth, predicted_probs)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth, predicted_probs)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    return auc_roc

