from collections import Counter

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

sns.set(style="whitegrid")


def read_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        dataset = [line.replace("\n", "") for line in file.readlines()]

    return dataset


if __name__ == '__main__':
    pos_dataset = read_data("./data/waimai_pos.txt")
    neg_dataset = read_data("./data/waimai_neg.txt")
    length_list = []
    comma_count = 0
    length_threshold = 126
    new_pos_dataset, new_neg_dataset = [], []

    for data in pos_dataset:
        if len(data) <= length_threshold:
            new_pos_dataset.append(data)

    for data in neg_dataset:
        if len(data) <= length_threshold:
            new_neg_dataset.append(data)

    # length_df = pd.DataFrame(data=[{"length": length, "count": count} for length, count in Counter(length_list).items()])
    # sns.barplot(x="length", y="count", data=length_df)
    # plt.show()
    # print(max(length_list))
    pos_train, pos_test = train_test_split(new_pos_dataset[:7800], train_size=0.8, random_state=7)
    neg_train, neg_test = train_test_split(new_neg_dataset[:7800], train_size=0.8, random_state=7)
    pos_train_, pos_dev = train_test_split(pos_train, train_size=0.8, random_state=7)
    neg_train_, neg_dev = train_test_split(neg_train, train_size=0.8, random_state=7)

    train_df = pd.DataFrame(data={
        "data": pos_train_ + neg_train_,
        "label": ["1"] * len(pos_train_) + ["0"] * len(neg_train_)
    })
    dev_df = pd.DataFrame(data={
        "data": pos_dev + neg_dev,
        "label": ["1"] * len(pos_dev) + ["0"] * len(neg_dev)
    })
    test_df = pd.DataFrame(data={
        "data": pos_test + neg_test,
        "label": ["1"] * len(pos_test) + ["0"] * len(neg_test)
    })
    train_df = shuffle(train_df)
    train_df.to_csv("./data/train.csv", index=False, header=None, sep="&")
    dev_df.to_csv("./data/dev.csv", index=False, header=None, sep="&")
    test_df.to_csv("./data/test.csv", index=False, header=None, sep="&")
