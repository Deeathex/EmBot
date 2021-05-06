class Emotions_Dataset(Dataset):

    def __init__(self, filename, maxlen, tokenizer, label_dict):
        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter=';')
        # name columns
        self.df.columns = ['sentence', 'emotion']
        # Initialize the tokenizer for the desired transformer model
        self.df['emotion'] = self.df['emotion'].map(label_dict)
        self.tokenizer = tokenizer
        # Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Select the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'emotion']
        # Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']
            # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()

        label = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label