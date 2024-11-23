import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
import os
class DataHandler:
	def __init__(self, train_dataset, test_dataset):
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def readTriplets(self):

		# train
		file_train_name = f'graph_train_{args.dataset}.csv'
		if not os.path.exists(file_train_name):
			inter_feat = self.train_dataset.dataset.inter_feat
			data_item = {
				'user_id': [],
				'relation': [],
				'item_id': []
			}
			"""
			disease: 1 - 1698
			examination: 1699 - 2188
			advanced_examination: 2189 - 2400
			Diagnosed_disease: 2400 - 2809
			"""
			id_token = self.train_dataset.dataset.field2id_token['item_id']
			item_num = self.train_dataset.dataset.item_num - 1

			for line in range(len(inter_feat)):
				for i in inter_feat.interaction['item_id_list'][line]:
					if i != 0:
						if 1 <= int(id_token[i]) and int(id_token[i]) <= 1698:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(0)
							data_item['item_id'].append(int(i))
						elif 1699 <= int(id_token[i]) and int(id_token[i]) <= 2188:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(1)
							data_item['item_id'].append(int(i))
						elif 2189 <= int(id_token[i]) and int(id_token[i]) <= 2400:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(2)
							data_item['item_id'].append(int(i))
						else:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(3)
							data_item['item_id'].append(int(i))

				data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
				data_item['relation'].append(5)
				data_item['item_id'].append(int(inter_feat.interaction['age_id'][line]) + item_num + 2)

				data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
				data_item['relation'].append(4)
				data_item['item_id'].append(int(inter_feat.interaction['gender_id'][line]) + item_num)

			can_triplets_np_train = pd.DataFrame()
			for key, values in data_item.items():
				can_triplets_np_train[key] = values

			can_triplets_np_train.to_csv(f'graph_test_{args.dataset}.csv', index=False)
		else:
			can_triplets_np_train = pd.read_csv(file_train_name)

		args.n_item_all = self.train_dataset.dataset.item_num + len(self.train_dataset.dataset.field2id_token['age_id']) + len(
			self.train_dataset.dataset.field2id_token['gender_id']) - 3

		args.n_user = self.train_dataset.dataset.user_num - 1
		args.n_age = len(self.train_dataset.dataset.field2id_token['age_id']) - 1
		args.n_gender = len(self.train_dataset.dataset.field2id_token['gender_id']) - 1

		can_triplets_np_train = np.unique(can_triplets_np_train, axis=0)
		inv_triplets_np_train = can_triplets_np_train.copy()
		inv_triplets_np_train[:, 0] = can_triplets_np_train[:, 2]
		inv_triplets_np_train[:, 2] = can_triplets_np_train[:, 0]
		inv_triplets_np_train[:, 1] = can_triplets_np_train[:, 1]
		triplets_train = np.concatenate((can_triplets_np_train, inv_triplets_np_train), axis=0)

		n_relations = max(triplets_train[:, 1])

		args.relation_num = n_relations

		# test
		file_test_name = f'graph_test_{args.dataset}.csv'
		if not os.path.exists(file_test_name):
			inter_feat = self.test_dataset.dataset.inter_feat
			data_item = {
				'user_id': [],
				'relation': [],
				'item_id': []
			}
			"""
            disease: 1 - 1698
            check: 1699 - 2188
            advanced_examination: 2189 - 2400
            Diagnosed_disease: 2400 - 2809
            """
			id_token = self.test_dataset.dataset.field2id_token['item_id']
			item_num = self.test_dataset.dataset.item_num - 1

			for line in range(len(inter_feat)):
				for i in inter_feat.interaction['item_id_list'][line]:
					if i != 0:
						if 1 <= int(id_token[i]) and int(id_token[i]) <= 1698:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(0)
							data_item['item_id'].append(int(i))
						elif 1699 <= int(id_token[i]) and int(id_token[i]) <= 2188:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(1)
							data_item['item_id'].append(int(i))
						elif 2189 <= int(id_token[i]) and int(id_token[i]) <= 2400:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(2)
							data_item['item_id'].append(int(i))
						else:
							data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
							data_item['relation'].append(3)
							data_item['item_id'].append(int(i))

				data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
				data_item['relation'].append(5)
				data_item['item_id'].append(int(inter_feat.interaction['age_id'][line]) + item_num + 2)

				data_item['user_id'].append(int(inter_feat.interaction['session_id'][line]))
				data_item['relation'].append(4)
				data_item['item_id'].append(int(inter_feat.interaction['gender_id'][line]) + item_num)

			can_triplets_np_test = pd.DataFrame()
			for key, values in data_item.items():
				can_triplets_np_test[key] = values

			can_triplets_np_test.to_csv(f'graph_test_{args.dataset}.csv', index=False)
		else:
			can_triplets_np_test = pd.read_csv(file_test_name)

		can_triplets_np_test = np.unique(can_triplets_np_test, axis=0)
		inv_triplets_np_test = can_triplets_np_test.copy()
		inv_triplets_np_test[:, 0] = can_triplets_np_test[:, 2]
		inv_triplets_np_test[:, 2] = can_triplets_np_test[:, 0]
		inv_triplets_np_test[:, 1] = can_triplets_np_test[:, 1]
		triplets_test = np.concatenate((can_triplets_np_test, inv_triplets_np_test), axis=0)

		return triplets_train, can_triplets_np_train, triplets_test, can_triplets_np_test
	
	def buildGraphs(self, triplets):
		kg_dict = defaultdict(list)
		kg_edges = list()

		print("Begin to load knowledge graph triples ...")

		kg_counter_dict = {}

		for h_id, r_id, t_id in tqdm(triplets, ascii=True):
			if h_id not in kg_counter_dict.keys():
				kg_counter_dict[h_id] = set()
			if t_id not in kg_counter_dict[h_id]:
				kg_counter_dict[h_id].add(t_id)
			else:
				continue
			kg_edges.append([h_id, t_id, r_id])
			kg_dict[h_id].append((r_id, t_id))

		return kg_edges, kg_dict
	
	def buildGraphMatrix(self,uiInfer):

		column_to_remove = uiInfer.shape[1] // 2  # 中间列的索引位置
		train_list = np.hstack((uiInfer[:, :column_to_remove], uiInfer[:, column_to_remove + 1:]))

		train_dict = {}

		uid_max = 0
		iid_max = 0
		for uid, iid in train_list:
			if uid-1 not in train_dict:
				train_dict[uid-1] = []
			train_dict[uid-1].append(iid-1)
			if uid-1 > uid_max:
				uid_max = uid-1
			if iid-1 > iid_max:
				iid_max = iid-1

		train_list = []
		for uid in train_dict:
			for iid in train_dict[uid]:
				train_list.append([uid, iid])
		train_list = np.array(train_list)

		graphMatrix = sp.csr_matrix((np.ones_like(train_list[:, 0]),(train_list[:, 0], train_list[:, 1])), dtype='float64',shape=(args.n_user, args.n_item_all))

		return graphMatrix

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def RelationDictBuild(self,uiInfer):
		relation_dict = {}
		with np.nditer(uiInfer, flags=['multi_index'], op_flags=['readwrite']) as it:
			for x in it:
				row_index, col_index = it.multi_index

				key = uiInfer[row_index, 0]

				nested_key = uiInfer[row_index, 2]

				nested_value = uiInfer[row_index, 1]

				if key not in relation_dict:
					relation_dict[key] = {}

				relation_dict[key][nested_key] = nested_value

		return relation_dict

	def buildUIMatrix(self, mat):
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def LoadData(self):

		kg_triplets_train, uiInfer_train, kg_triplets_test, uiInfer_test  = self.readTriplets()

		self.uiInfer_train = uiInfer_train
		self.uiInfer_test = uiInfer_test
		self.kg_matrix_train = self.buildGraphMatrix(uiInfer_train)
		self.kg_matrix_test = self.buildGraphMatrix(uiInfer_test)
		
		self.diffusionData_train = DiffusionData(torch.FloatTensor(self.kg_matrix_train.A))
		self.diffusionData_test = DiffusionData(torch.FloatTensor(self.kg_matrix_test.A))

		self.diffusionLoader_train = dataloader.DataLoader(self.diffusionData_train, batch_size=args.batch, shuffle=True, num_workers=0)
		self.diffusionLoader_test = dataloader.DataLoader(self.diffusionData_test, batch_size=args.batch, shuffle=True,
													 num_workers=0)

		self.relation_dict_train = self.RelationDictBuild(uiInfer_train)
		self.relation_dict_test = self.RelationDictBuild(uiInfer_test)


class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data
	
	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)