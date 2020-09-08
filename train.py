from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import torch
torch.cuda.empty_cache()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
dataset_path = 'leetData.csv'


# Read the dataset

train_batch_size = 4
num_epochs = 4
model_save_path = 'output/Leetcode_continue_training-'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Load a pre-trained sentence transformer model
model = SentenceTransformer('model')

# Convert the dataset to a DataLoader ready for training
logging.info("Reading train dataset")

train_samples = []
dev_samples = []
test_samples = []


with open('leetData.csv', 'rt', encoding='utf8') as f:
    reader = csv.reader(f)
    for row in reader:
            
        score = float(row[2]) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row[0], row[1]], label=score)
        if len(row)== 4:
            if row[3] == "dev":
                dev_samples.append(inp_example)
            else:
                test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)


train_dataset = SentencesDataset(train_samples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Reading dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. 
warmup_steps = math.ceil(len(train_dataset)* num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=100,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on the test set
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='LeetCode test')
test_evaluator(model, output_path=model_save_path)

