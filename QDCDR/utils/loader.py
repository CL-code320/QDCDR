"""
Data loader for TACRED json files.
"""

import json
from optparse import Values
import random
import torch
import numpy as np
import codecs
from pathlib import Path
import copy

current_path = Path(__file__).resolve()
dataset_path = current_path.parent.parent.parent / "dataset"


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, evaluation):
        self.batch_size = batch_size
        self.opt = opt
        self.eval = evaluation

        # ************* source data *****************
        source_train_data = dataset_path / filename / "train.txt"
        source_test_data = dataset_path / filename / "test.txt"
        self.source_ma_set, self.source_ma_list, self.source_train_data, self.source_test_data, self.source_user, self.source_item = self.read_data(source_train_data, source_test_data)
        print(f"Max source_pos_item: {max(max(i) for i in self.source_ma_list.values())}, Min source_pos_item: {min(min(i) for i in self.source_ma_list.values())}")
        opt["source_user_num"] = len(self.source_user)
        opt["source_item_num"] = len(self.source_item)

        print("source_user_num", opt["source_user_num"])
        print("source_item_num", opt["source_item_num"])

        # ************* target data *****************
        filename = filename.split("_")
        filename = filename[1] + "_" + filename[0]
        target_train_data = dataset_path / filename / "train.txt"
        target_test_data = dataset_path / filename / "test.txt"
        self.target_ma_set, self.target_ma_list, self.target_train_data, self.target_test_data, self.target_user, self.target_item = self.read_data(target_train_data, target_test_data)
        opt["target_user_num"] = len(self.target_user)
        opt["target_item_num"] = len(self.target_item)

        print("target_user_num", opt["target_user_num"])
        print("target_item_num", opt["target_item_num"])

        # ************* shared domain data *************
        self.shared_ma_set, self.shared_ma_list,  self.shared_train_data, self.shared_test_data = self.create_shared_data(opt)
        opt["share_user_num"] = len(self.source_user)  # source 和 target 用户是相同的
        opt["share_item_num"] = opt["source_item_num"] + opt["target_item_num"]

        opt["rate"] = self.rate()

        assert opt["source_user_num"] == opt["target_user_num"]
        if evaluation == -1:
            data = self.preprocess()
        else :
            data = self.preprocess_for_predict()
        # shuffle for training
        if evaluation == -1:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            if batch_size > len(data):
                batch_size = len(data)
                self.batch_size = batch_size
            if len(data)%batch_size != 0:
                data += data[:batch_size]
            data = data[: (len(data)//batch_size) * batch_size]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def create_shared_data(self, opt):
        """
        Create shared domain data by merging source and target data with item ID offset.
        """
        shared_train_data = []
        shared_test_data = []

        # Merge source domain training data directly
        for user, item in self.source_train_data:
            shared_train_data.append([user, item])  # Source data is added as is
        # Merge target domain training data with item ID offset
        for user, item in self.target_train_data:
            shared_train_data.append([user, item + opt["source_item_num"]])  # Offset target item IDs

        # Merge source domain test data directly
        for user, item_list in self.source_test_data:
            shared_test_data.append([user, item_list])  # Source test data is added as is
        # Merge target domain test data with item ID offset for each item
        for user, item_list in self.target_test_data:
            shifted_item_list = [item + opt["source_item_num"] for item in item_list]  # Offset each target item ID
            shared_test_data.append([user, shifted_item_list])

        # Deep copy source ma_set and ma_list to avoid reference issues
        shared_ma_set = copy.deepcopy(self.source_ma_set)
        shared_ma_list = copy.deepcopy(self.source_ma_list)
        # Merge target ma_set and ma_list, offsetting item IDs
        for user, items in self.target_ma_set.items():
            if user not in shared_ma_set:
                shared_ma_set[user] = set()
                shared_ma_list[user] = []
            shared_ma_set[user].update(items)
            # Offset each target item ID before extending the list
            shared_ma_list[user].extend([item + opt["source_item_num"] for item in items])

        return shared_ma_set, shared_ma_list, shared_train_data, shared_test_data

    

    def read_data(self, train_file, test_file):
        with codecs.open(train_file, "r", encoding="utf-8") as infile:
            train_data = []
            user = {}
            item = {}
            ma = {}
            ma_list = {}
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                # Use more pythonic way to check and assign new user/item IDs
                if line[0] not in user:
                    user[line[0]] = len(user)
                if line[1] not in item:
                    item[line[1]] = len(item)
                line[0] = user[line[0]]
                line[1] = item[line[1]]
                train_data.append([line[0], line[1]])
                if line[0] not in ma:
                    ma[line[0]] = set()
                    ma_list[line[0]] = []
                ma[line[0]].add(line[1])
                ma_list[line[0]].append(line[1])
        with codecs.open(test_file, "r", encoding="utf-8") as infile:
            test_data = []
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                # Skip if user or item not in training set
                if line[0] not in user:
                    continue
                if line[1] not in item:
                    continue
                line[0] = user[line[0]]
                line[1] = item[line[1]]

                ret = [line[1]]
                for i in range(999):
                    while True:
                        rand = random.randint(0, len(item) - 1)
                        if rand in ma[line[0]]:
                            continue
                        ret.append(rand)
                        break
                test_data.append([line[0], ret])

        return ma, ma_list, train_data, test_data, user, item

    def rate(self):
        ret = []
        for i in range(len(self.source_ma_set)):
            ret = len(self.source_ma_set[i]) / (len(self.source_ma_set[i]) + len(self.target_ma_set[i]))
        return ret

    def preprocess_for_predict(self):
        processed=[]
        if self.eval == 1:
            for d in self.source_test_data:
                processed.append([d[0],d[1]]) # user, item_list(pos in the first node)
        else :
            for d in self.target_test_data:
                processed.append([d[0],d[1]]) # user, item_list(pos in the first node)
        return processed
    def preprocess(self):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in self.source_train_data:
            d = [d[1], d[0]]
            processed.append(d + [-1])
        for d in self.target_train_data:
            processed.append([-1] + d)
        for d in self.shared_train_data:
            d = [d[1], d[0]]
            processed.append(d + [-2])
        return processed

    def find_pos(self,ma_list, user):
        rand = random.randint(0, 1000000)
        rand %= len(ma_list[user])
        return ma_list[user][rand]

    def find_neg(self, ma_set, user, type):
        n = 5
        while n:
            n -= 1
            rand = random.randint(0, self.opt[type] - 1)
            if rand not in ma_set[user]:
                return rand
        return rand

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        if self.eval!=-1 :
            batch = list(zip(*batch))
            return (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]))

        else :
            source_neg_tmp = []
            target_neg_tmp = []
            source_pos_tmp = []
            target_pos_tmp = []
            share_neg_tmp = []
            share_pos_tmp = []
            user = []
            for b in batch:
                if b[0] == -1:
                    source_pos_tmp.append(self.find_pos(self.source_ma_list, b[1]))
                    target_pos_tmp.append(b[2])
                    share_pos_tmp.append(self.find_pos(self.shared_ma_list, b[1]))
                elif b[2] == -1:
                    source_pos_tmp.append(b[0])
                    target_pos_tmp.append(self.find_pos(self.target_ma_list, b[1]))
                    share_pos_tmp.append(self.find_pos(self.shared_ma_list, b[1]))
                else:
                    source_pos_tmp.append(self.find_pos(self.source_ma_list, b[1]))
                    target_pos_tmp.append(self.find_pos(self.target_ma_list, b[1]))
                    share_pos_tmp.append(b[0])
            
                source_neg_tmp.append(self.find_neg(self.source_ma_set, b[1], "source_item_num"))
                target_neg_tmp.append(self.find_neg(self.target_ma_set, b[1], "target_item_num"))
                share_neg_tmp.append(self.find_neg(self.shared_ma_set, b[1], "share_item_num"))
                user.append(b[1])
            return (torch.LongTensor(user), torch.LongTensor(source_pos_tmp), torch.LongTensor(source_neg_tmp), torch.LongTensor(target_pos_tmp), torch.LongTensor(target_neg_tmp), torch.LongTensor(share_pos_tmp), torch.LongTensor(share_neg_tmp))
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)