from sentiment_analysis.utils.Utils import Utils

import matplotlib.pyplot as plt
import seaborn as sns

from sentiment_analysis.utils.constants import LENGTH, CARER_DATASET


class ExploratoryDataAnalysisModule:
    def __init__(self, path_to_dataset):
        sns.set()
        self.train_dataframe = Utils.load_data_frame(path=path_to_dataset + '/train.txt', separator='|')
        self.validation_dataframe = Utils.load_data_frame(path=path_to_dataset + '/val.txt', separator='|')

    def plot_distribution_of_emotions(self):
        print(self.train_dataframe.head())
        sns.countplot(self.train_dataframe.emotion)
        plt.show()
        print(self.validation_dataframe.head())
        sns.countplot(self.validation_dataframe.emotion)
        plt.show()

    def plot_length_of_train_data(self):
        self.train_dataframe[LENGTH] = self.train_dataframe.sentence.apply(lambda x: len(x))
        plt.plot(self.train_dataframe.length)
        plt.show()
        print('Max length of our text body: ', self.train_dataframe.length.max())

    def plot_pie_chart_emotions(self):
        self.__plot_data_distribution_as_pie_chart(self.train_dataframe)
        self.__plot_data_distribution_as_pie_chart(self.validation_dataframe)

    def __plot_data_distribution_as_pie_chart(self, data_frame):
        joy_count = data_frame[data_frame['emotion'] == 'joy'].size
        love_count = data_frame[data_frame['emotion'] == 'love'].size
        sadness_count = data_frame[data_frame['emotion'] == 'sadness'].size
        anger_count = data_frame[data_frame['emotion'] == 'anger'].size
        fear_count = data_frame[data_frame['emotion'] == 'fear'].size
        surprise_count = data_frame[data_frame['emotion'] == 'surprise'].size

        labels = 'joy', 'love', 'sadness', 'anger', 'fear', 'surprise'
        sizes = [joy_count, love_count, sadness_count, anger_count, fear_count, surprise_count]
        explode = (0.1, 0, 0, 0, 0, 0)  # only "explode" the first slice

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()


eda = ExploratoryDataAnalysisModule(CARER_DATASET)
eda.plot_pie_chart_emotions()
