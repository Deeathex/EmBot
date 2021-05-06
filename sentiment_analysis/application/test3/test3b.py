import numpy as np
import pandas as pd
import os

MODEL_OUT_DIR = '../ai_models/bert_emotion'
TRAIN_FILE_PATH = '../dataset/emotions_dataset/train.txt'
VALID_FILE_PATH = '../dataset/emotions_dataset/val.txt'
TEST_FILE_PATH = '../dataset/emotions_dataset/test.txt'
## Model Configurations
MAX_LEN_TRAIN = 68
MAX_LEN_VALID = 68
MAX_LEN_TEST = 68
BATCH_SIZE = 160
LR = 1e-5
NUM_EPOCHS = 10
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'
LABEL_DICT = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}

if not os.path.isdir(MODEL_OUT_DIR):
    os.makedirs(MODEL_OUT_DIR)

# class Emotions_Dataset(Dataset):
#
#     def __init__(self, filename, maxlen, tokenizer, label_dict):
#         # Store the contents of the file in a pandas dataframe
#         self.df = pd.read_csv(filename, delimiter=';')
#         # name columns
#         self.df.columns = ['sentence', 'emotion']
#         # Initialize the tokenizer for the desired transformer model
#         self.df['emotion'] = self.df['emotion'].map(label_dict)
#         self.tokenizer = tokenizer
#         # Maximum length of the tokens list to keep all the sequences of fixed size
#         self.maxlen = maxlen
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, index):
#         # Select the sentence and label at the specified index in the data frame
#         sentence = self.df.loc[index, 'sentence']
#         label = self.df.loc[index, 'emotion']
#         # Preprocess the text to be suitable for the transformer
#         tokens = self.tokenizer.tokenize(sentence)
#         tokens = ['[CLS]'] + tokens + ['[SEP]']
#         if len(tokens) < self.maxlen:
#             tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
#         else:
#             tokens = tokens[:self.maxlen - 1] + ['[SEP]']
#             # Obtain the indices of the tokens in the BERT Vocabulary
#         input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#         input_ids = torch.tensor(input_ids)
#         # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
#         attention_mask = (input_ids != 0).long()
#
#         label = torch.tensor(label, dtype=torch.long)
#
#         return input_ids, attention_mask, label
#
# class BertEmotionClassifier(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         #The classification layer that takes the [CLS] representation and outputs the logit
#         self.cls_layer = nn.Linear(config.hidden_size, 6)
#
#     def forward(self, input_ids, attention_mask):
#         #Feed the input to Bert model to obtain contextualized representations
#         reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         #Obtain the representations of [CLS] heads
#         cls_reps = reps[:, 0]
#         logits = self.cls_layer(cls_reps)
#         return logits