import pandas as pd
import matplotlib.pyplot as plt


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def load_csv_file_as_data_frame(path_to_csv, separator=','):
        return pd.read_csv(path_to_csv, sep=separator)

    @staticmethod
    def write_data_frame_to_csv(data_frame, csv_path):
        data_frame.to_csv(r'' + csv_path, index=False, header=True)

    @staticmethod
    def plot_data_distribution(data_frame):
        joy_count = data_frame[data_frame['emotion'] == 'joy'].size
        shame_count = data_frame[data_frame['emotion'] == 'shame'].size
        anger_count = data_frame[data_frame['emotion'] == 'anger'].size
        disgust_count = data_frame[data_frame['emotion'] == 'disgust'].size
        sadness_count = data_frame[data_frame['emotion'] == 'sadness'].size
        guilt_count = data_frame[data_frame['emotion'] == 'guilt'].size
        fear_count = data_frame[data_frame['emotion'] == 'fear'].size

        labels = 'joy', 'shame', 'anger', 'disgust', 'sadness', 'guilt', 'fear'
        sizes = [joy_count, shame_count, anger_count, disgust_count, sadness_count, guilt_count, fear_count]
        explode = (0.1, 0, 0, 0, 0, 0, 0)  # only "explode" the first slice

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()
        # fig = plt.figure()
        # ax = fig.add_axes([0, 0, 1, 1])
        # emotions = ['joy', 'shame', 'anger', 'disgust', 'sadness', 'guilt', 'fear']
        # emotions_count = [joy_count, shame_count, anger_count, disgust_count, sadness_count, guilt_count, fear_count]
        # ax.bar(emotions, emotions_count)
        # plt.show()
