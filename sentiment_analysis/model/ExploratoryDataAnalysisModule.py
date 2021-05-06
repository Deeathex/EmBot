from sentiment_analysis.utils.Utils import Utils

import matplotlib.pyplot as plt
import seaborn as sns

from sentiment_analysis.utils.constants import LENGTH


class ExploratoryDataAnalysisModule:
    def __init__(self, path_to_dataset):
        sns.set()
        self.train_dataframe = Utils.load_data_frame(path=path_to_dataset + '/train.txt', separator='|')
        self.validation_dataframe = Utils.load_data_frame(path=path_to_dataset + '/val.txt', separator='|')

    def plot_distribution_of_emotions(self):
        sns.countplot(self.train_dataframe.emotion)
        plt.show()

    def plot_length_of_train_data(self):
        self.train_dataframe[LENGTH] = self.train_dataframe.sentence.apply(lambda x: len(x))
        plt.plot(self.train_dataframe.length)
        plt.show()
        print('Max length of our text body: ', self.train_dataframe.length.max())
