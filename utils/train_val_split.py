import os
import sys
import numpy as np
import pandas as pd


def train_val_split():
    full_imgs = np.array(os.listdir('../train_shuffle'))
    full_labels = pd.read_csv('../Released_Data/train_data.csv').to_numpy()

    labels_list = [set(),set(),set()]

    #prepare lists based on the super classes
    labels_list = [{}, {}, {}]
    val_labels_list = [{}, {}, {}]

    for i in range(len(full_labels)):
        if full_labels[i][2] in labels_list[full_labels[i][1]]:
            labels_list[full_labels[i][1]][full_labels[i][2]] += 1
        else:
            labels_list[full_labels[i][1]][full_labels[i][2]] = 1
            val_labels_list[full_labels[i][1]][full_labels[i][2]] = 0

    # Check the number of times each sub class appears
    print(dict(sorted(labels_list[0].items(), key=lambda item: item[1])))
    print(dict(sorted(labels_list[1].items(), key=lambda item: item[1])))
    print(dict(sorted(labels_list[2].items(), key=lambda item: item[1])))

    #Splitting the train and validation datasets (90 - 10 split)
    train_imgs = []
    val_imgs = []

    for i in range(len(full_labels)):
        if val_labels_list[full_labels[i][1]][full_labels[i][2]] <= (0.1)*(labels_list[full_labels[i][1]][full_labels[i][2]]):
            val_labels_list[full_labels[i][1]][full_labels[i][2]] += 1
            val_imgs.append(full_labels[i][0])
        else:
            train_imgs.append(full_labels[i][0])

    #Check if there is any common image between training and validation datasets
    common_names = [name for name in train_imgs if name in val_imgs]
    print("Common names: ", common_names)

    train_data = full_labels[np.isin(full_labels[:, 0], train_imgs)]
    val_data = full_labels[np.isin(full_labels[:, 0], val_imgs)]

    #Check if there is any name missing from the training dataset
    ver_train_names = train_data[:, 0]
    error_names = [name for name in ver_train_names if name not in train_imgs]
    print("Error names: ", error_names)

    print("Total number of images: ", len(full_imgs))
    print("Number of training images: ", train_data.shape)
    print("Number of validation images: ", val_data.shape)
    # print(type(train_data[0][0]))
    # print(type(train_data[0][1]))
    # print(type(val_data[0][0]))
    # print(type(val_data[0][1]))


    # --------Code to check the distribution of super and sub classes in the training and validation datasets---------
    # train_sub_dict = {}
    # train_super_dict = {}
    # val_sub_dict = {}
    # val_super_dict = {}

    # for i in range(full_labels.shape[0]):
    #     name = full_labels[i][0]
    #     super_class = full_labels[i][1]
    #     sub_class = full_labels[i][2]

    #     if name in val_data[:, 0]:
    #         if super_class in val_super_dict:
    #             val_super_dict[super_class] += 1
    #         else:
    #             val_super_dict[super_class] = 1
            
    #         if sub_class in val_sub_dict:
    #             val_sub_dict[sub_class] += 1
    #         else:
    #             val_sub_dict[sub_class] = 1
    #     else:
    #         if super_class in train_super_dict:
    #             train_super_dict[super_class] += 1
    #         else:
    #             train_super_dict[super_class] = 1
            
    #         if sub_class in train_sub_dict:
    #             train_sub_dict[sub_class] += 1
    #         else:
    #             train_sub_dict[sub_class] = 1

    # print("Super Class distribution:")
    # for k in val_super_dict:
    #     print("Class: ", k, " Validation: ", val_super_dict[k], " Train: ", train_super_dict[k], " Ratio: ", train_super_dict[k]/val_super_dict[k])

    # print("------------------------------")
    # print("Sub Class distribution:")
    # min_val = 10000
    # max_val = -1
    # for k in val_sub_dict:
    #     min_val = min(min_val, train_sub_dict[k]/val_sub_dict[k])
    #     max_val = max(max_val, train_sub_dict[k]/val_sub_dict[k])
    #     print("Class: ", k, " Validation: ", val_sub_dict[k], " Train: ", train_sub_dict[k], " Ratio: ", train_sub_dict[k]/val_sub_dict[k])

    # print("Minimum Sub Class ratio: ", min_val)
    # print("Maximum Sub Class ratio: ", max_val)
