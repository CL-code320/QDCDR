import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker(object):
    def __init__(self, opt, filename):
        self.opt = opt
        self.user = set() #集合可以起到一个去重的作用
        self.item = set()
        user_map = {}
        item_map = {}
        data=[]
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if user_map.get(line[0], "zxczxc") == "zxczxc":
                    user_map[line[0]] = len(user_map) #user真实id到新id的映射
                if item_map.get(line[1], "zxczxc") == "zxczxc":
                    item_map[line[1]] = len(item_map) #item真实id到新id的映射
                line[0] = user_map[line[0]]
                line[1] = item_map[line[1]]
                data.append((int(line[0]),int(line[1]),float(line[2]))) #data里是由新id组成的
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        opt["number_user"] = len(self.user)
        opt["number_item"] = len(self.item)

        print("number_user", len(self.user))
        print("number_item", len(self.item))
        
        self.raw_data = data
        self.UV,self.VU, self.adj = self.preprocess( data, opt)

    def preprocess(self,data,opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        real_adj = {}

        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            if edge[0] not in user_real_dict.keys():
                user_real_dict[edge[0]] = set()
            user_real_dict[edge[0]].add(edge[1])

            VU_edges.append([edge[1], edge[0]])
            if edge[1] not in item_real_dict.keys():
                item_real_dict[edge[1]] = set()
            item_real_dict[edge[1]].add(edge[0])

            all_edges.append([edge[0],edge[1] + opt["number_user"]])
            all_edges.append([edge[1] + opt["number_user"], edge[0]])
            if edge[0] not in real_adj :
                real_adj[edge[0]] = {}
            real_adj[edge[0]][edge[1]] = 1

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_user"], opt["number_item"]),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_item"], opt["number_user"]),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(opt["number_item"]+opt["number_user"], opt["number_item"]+opt["number_user"]),dtype=np.float32)
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

        print("real graph loaded!")
        return UV_adj, VU_adj, all_adj




# 共享图构建器
class SharedGraphMaker(object):
    def __init__(self, opt, filename_A, filename_B):
        self.opt = opt
        self.user = set()   # 用户集合
        self.item_A = set() # A领域物品集合
        self.item_B = set() # B领域物品集合
        
        self.user_map = {}  # 用户ID重编码映射
        self.item_map_A = {}  # A领域物品ID重编码映射
        self.item_map_B = {}  # B领域物品ID重编码映射
        
        self.data_A = []  # A领域的数据
        self.data_B = []  # B领域的数据
        
        # 读取A领域数据
        self._load_data(filename_A, "A")
        
        # 读取B领域数据
        self._load_data(filename_B, "B")
        
        opt["number_user"] = len(self.user)
        opt["number_item_A"] = len(self.item_map_A)  # A领域物品数量
        opt["number_item_B"] = len(self.item_map_B)  # B领域物品数量
        
        print("number_user", len(self.user))
        print("number_item_A", len(self.item_map_A))
        print("number_item_B", len(self.item_map_B))

        # 预处理图数据
        self.UV, self.VU, self.adj = self.preprocess(self.data_A, self.data_B, opt)

    def _load_data(self, filename, domain):
        """加载A或B领域的数据"""
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])  # 用户ID
                line[1] = int(line[1])  # 物品ID

                # 用户重编码
                if line[0] not in self.user_map:
                    self.user_map[line[0]] = len(self.user_map)

                # A领域物品映射
                if domain == "A" and line[1] not in self.item_map_A:
                    self.item_map_A[line[1]] = len(self.item_map_A)
                    self.item_A.add(line[1])
                # B领域物品映射
                if domain == "B" and line[1] not in self.item_map_B:
                    self.item_map_B[line[1]] = len(self.item_map_B)
                    self.item_B.add(line[1])
                
                # 更新数据（A和B领域分别处理）
                if domain == "A":
                    line[1] = self.item_map_A[line[1]]  # A领域物品ID
                    self.data_A.append((self.user_map[line[0]], line[1], 1))  # 1表示交互发生
                else:
                    line[1] = self.item_map_B[line[1]]  # B领域物品ID
                    self.data_B.append((self.user_map[line[0]], line[1], 1))  # 1表示交互发生
                self.user.add(self.user_map[line[0]])

    def preprocess(self, data_A, data_B, opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        
        # 处理A领域的数据
        for edge in data_A:
            UV_edges.append([edge[0], edge[1]])
            VU_edges.append([edge[1], edge[0]])
            all_edges.append([edge[0], edge[1] + opt["number_user"]])  # B领域物品ID的偏移
            all_edges.append([edge[1] + opt["number_user"], edge[0]])

        # 处理B领域的数据
        for edge in data_B:
            UV_edges.append([edge[0], edge[1] + opt["number_item_A"]])  # B领域物品ID加上A领域物品数量偏移
            VU_edges.append([edge[1] + opt["number_item_A"], edge[0]])  # B领域物品ID加上偏移
            all_edges.append([edge[0], edge[1] + opt["number_user"] +opt["number_item_A"]])  # B领域物品ID的偏移
            all_edges.append([edge[1] + opt["number_user"] + opt["number_item_A"], edge[0]])

        # 转换为Numpy数组
        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)

        # 创建邻接矩阵
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_user"], opt["number_item_A"] + opt["number_item_B"]),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_item_A"] + opt["number_item_B"], opt["number_user"]),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),
                                shape=(opt["number_item_A"] + opt["number_item_B"] + opt["number_user"],
                                       opt["number_item_A"] + opt["number_item_B"] + opt["number_user"]),
                                dtype=np.float32)

        # 规范化邻接矩阵
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)

        # 转换为稀疏张量
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

        print("real graph loaded!")
        return UV_adj, VU_adj, all_adj


