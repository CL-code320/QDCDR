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
        self.user = set()  # Set for unique user IDs
        self.item = set()  # Set for unique item IDs
        user_map = {}      # Mapping from original user ID to new user ID
        item_map = {}      # Mapping from original item ID to new item ID
        data = []
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                # Assign new user ID if not already mapped
                if line[0] not in user_map:
                    user_map[line[0]] = len(user_map)
                # Assign new item ID if not already mapped
                if line[1] not in item_map:
                    item_map[line[1]] = len(item_map)
                line[0] = user_map[line[0]]
                line[1] = item_map[line[1]]
                # Store the remapped user, item, and rating
                data.append((int(line[0]), int(line[1]), float(line[2])))
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        opt["number_user"] = len(self.user)
        opt["number_item"] = len(self.item)

        print("number_user", len(self.user))
        print("number_item", len(self.item))
        
        self.raw_data = data
        self.UV, self.VU, self.adj = self.preprocess(data, opt)

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





class SharedGraphMaker(object):
    def __init__(self, opt, filename_A, filename_B):
        self.opt = opt
        self.user = set()   # Set for unique user IDs
        self.item_A = set() # Set for unique item IDs in domain A
        self.item_B = set() # Set for unique item IDs in domain B
        
        self.user_map = {}      # Mapping from original user ID to new user ID
        self.item_map_A = {}    # Mapping from original item ID to new item ID in domain A
        self.item_map_B = {}    # Mapping from original item ID to new item ID in domain B
        
        self.data_A = []  # Data for domain A
        self.data_B = []  # Data for domain B
        
        # Load data for domain A
        self._load_data(filename_A, "A")
        # Load data for domain B
        self._load_data(filename_B, "B")
        
        # Store the number of users and items in the options dictionary
        opt["number_user"] = len(self.user)
        opt["number_item_A"] = len(self.item_map_A)
        opt["number_item_B"] = len(self.item_map_B)
        
        print("number_user", len(self.user))
        print("number_item_A", len(self.item_map_A))
        print("number_item_B", len(self.item_map_B))

        # Preprocess the graph data to generate adjacency matrices
        self.UV, self.VU, self.adj = self.preprocess(self.data_A, self.data_B, opt)

    def _load_data(self, filename, domain):
        """
        Load data from a file for a specific domain (A or B).
        Each line is expected to be: user_id \t item_id \t [rating/interaction]
        """
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])  # User ID
                line[1] = int(line[1])  # Item ID

                # Assign new user ID if not already mapped
                if line[0] not in self.user_map:
                    self.user_map[line[0]] = len(self.user_map)

                # Assign new item ID for domain A if not already mapped
                if domain == "A" and line[1] not in self.item_map_A:
                    self.item_map_A[line[1]] = len(self.item_map_A)
                    self.item_A.add(line[1])
                # Assign new item ID for domain B if not already mapped
                if domain == "B" and line[1] not in self.item_map_B:
                    self.item_map_B[line[1]] = len(self.item_map_B)
                    self.item_B.add(line[1])
                
                # Update data for the corresponding domain
                if domain == "A":
                    line[1] = self.item_map_A[line[1]]  # Remap item ID for domain A
                    self.data_A.append((self.user_map[line[0]], line[1], 1))  # 1 indicates an interaction
                else:
                    line[1] = self.item_map_B[line[1]]  # Remap item ID for domain B
                    self.data_B.append((self.user_map[line[0]], line[1], 1))  # 1 indicates an interaction
                self.user.add(self.user_map[line[0]])

    def preprocess(self, data_A, data_B, opt):
        """
        Build adjacency matrices for the shared graph from domain A and B data.
        Returns user-item, item-user, and full adjacency matrices as torch sparse tensors.
        """
        UV_edges = []    # Edges from user to item
        VU_edges = []    # Edges from item to user
        all_edges = []   # All edges for the full adjacency matrix
        
        # Process domain A data
        for edge in data_A:
            UV_edges.append([edge[0], edge[1]])
            VU_edges.append([edge[1], edge[0]])
            # Offset item indices by number of users for the full adjacency matrix
            all_edges.append([edge[0], edge[1] + opt["number_user"]])
            all_edges.append([edge[1] + opt["number_user"], edge[0]])

        # Process domain B data
        for edge in data_B:
            # Offset item indices by number of items in domain A for UV/VU matrices
            UV_edges.append([edge[0], edge[1] + opt["number_item_A"]])
            VU_edges.append([edge[1] + opt["number_item_A"], edge[0]])
            # Offset item indices by number of users and items in domain A for the full adjacency matrix
            all_edges.append([edge[0], edge[1] + opt["number_user"] + opt["number_item_A"]])
            all_edges.append([edge[1] + opt["number_user"] + opt["number_item_A"], edge[0]])

        # Convert edge lists to numpy arrays
        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)

        # Create sparse adjacency matrices
        UV_adj = sp.coo_matrix(
            (np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
            shape=(opt["number_user"], opt["number_item_A"] + opt["number_item_B"]),
            dtype=np.float32
        )
        VU_adj = sp.coo_matrix(
            (np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
            shape=(opt["number_item_A"] + opt["number_item_B"], opt["number_user"]),
            dtype=np.float32
        )
        all_adj = sp.coo_matrix(
            (np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),
            shape=(opt["number_item_A"] + opt["number_item_B"] + opt["number_user"],
                   opt["number_item_A"] + opt["number_item_B"] + opt["number_user"]),
            dtype=np.float32
        )

        # Normalize adjacency matrices
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)

        # Convert to torch sparse tensors
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

        print("real graph loaded!")
        return UV_adj, VU_adj, all_adj


