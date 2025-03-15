import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

class DataHandler:
	def __init__(self):
		# if args.data == 'yelp':
		# 	predir = 'Data/yelp/'
		# elif args.data == 'ml10m':
		# 	predir = 'Data/ml10m/'
		# elif args.data == 'tmall':
		# 	predir = 'Data/tmall/'
		# elif args.data == 'gowalla':
		# 	predir = 'Data/gowalla/'
		# elif args.data == 'amazon-book':
		# 	predir = 'Data/amazon-book/'
		if args.data == 'music':
			predir = 'Data/amazon-book/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

'''
加载 pkl 文件，并转换为 scipy.sparse.coo_matrix 类型的稀疏矩阵。
'''

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret
"""
计算归一化的邻接矩阵：

    计算度矩阵 degree = np.array(mat.sum(axis=-1))
    计算 D^{-0.5}（度矩阵的逆平方根），避免除零错误
    归一化邻接矩阵 D^{-0.5} * A * D^{-0.5}

"""
	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

"""
构造 PyTorch 张量的邻接矩阵：

    创建用户-物品的交互矩阵，并扩展成 UI 形式的邻接矩阵
    归一化
    转换为 PyTorch 的 FloatTensor 并移动到 CUDA 设备上

"""

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		# mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()
"""
生成用户-物品的采样索引矩阵：

    user_sample_idx：用户索引
    item_sample_idx：物品索引

"""
	def makeSample(self):
		user_sample_idx = t.tensor([[args.user + i for i in range(args.item)] * args.user])
		item_sample_idx = t.tensor([[i for i in range(args.user)] * args.item])
		return user_sample_idx, item_sample_idx
"""
生成 PyTorch 版的掩码矩阵 mask：

    u_u_mask（用户-用户）：全零矩阵
    u_i_mask（用户-物品）：全一矩阵
    i_i_mask（物品-物品）：全零矩阵
    i_u_mask（物品-用户）：全一矩阵
    拼接 u_mask 和 i_mask 形成最终的 mask


"""
	def makeMask(self):
		u_u_mask = t.zeros(size=(args.user, args.user), dtype=bool)
		u_i_mask = t.ones(size=(args.user, args.item), dtype=bool)

		i_i_mask = t.zeros(size=(args.item, args.item), dtype=bool)
		i_u_mask = t.ones(size=(args.item, args.user), dtype=bool)

		u_mask = t.concat([u_u_mask, u_i_mask], dim=-1)
		i_mask = t.concat([i_u_mask, i_i_mask], dim=-1)

		mask = t.concat([u_mask, i_mask], dim=0)
		return mask
'''
加载训练数据和测试数据
计算 user 和 item 数量
生成 PyTorch 版本的邻接矩阵 torchBiAdj
生成掩码矩阵 mask
使用 DataLoader 加载训练数据和测试数据

'''

	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		self.mask = self.makeMask()
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
"""
TrnMaskedData 类（带负样本的训练数据）
"""
class TrnMaskedData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])