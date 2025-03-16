
# For any questions, don't hesitate to contact me: Ding Zou (m202173662@hust.edu.cn)

import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.MCCLK import Recommender
from utils.evaluate import test
from utils.helper import early_stopping

import logging

import torch as t
import utils.TimeLogger as logger
from utils.TimeLogger import log
from Params import args
from Model import TransGNN
from DataHandler import DataHandler
import numpy as np
import pickle
from utils.Utils import *
import os
import setproctitle

###############transGNN部分


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            # if tstFlag:
            #     reses = self.testEpoch()
            #     log(self.makePrint('Test', ep, reses, tstFlag))
            #     self.saveHistory()
            # print()
        # reses = self.testEpoch()
        # log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def prepareModel(self):
        self.model = TransGNN().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch


        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            ########for mcclk
            global g_trans_u_embeddings, g_trans_i_embeddings

            bprLoss,g_trans_u_embeddings, g_trans_i_embeddings = self.model.calcLosses(ancs, poss, negs, self.handler.torchBiAdj)
            loss = bprLoss



            epLoss += loss.item()
            epPreLoss += bprLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('Step %d/%d: loss = %.3f         ' % (i, steps, loss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret,g_trans_u_embeddings, g_trans_i_embeddings

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall, epNdcg = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)

            allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('../History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, '../Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load('../Models/' + args.load_model + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('../History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

'''
这段代码实现了一个推荐系统的训练和评估过程，包含数据预处理、模型训练和指标评估。
'''

'''
将输入的用户-物品交互数据（train_entity_pairs）转为 PyTorch 张量，并将其分割为批次供模型训练或评估。

返回一个字典 feed_dict，包含用户 (users)、物品 (items) 和标签 (labels) 的张量。


'''
def get_feed_dict(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict

def get_feed_dict_topk(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict
    # def negative_sampling(user_item, train_user_set):
    #     neg_items = []
    #     for user, _ in user_item.cpu().numpy():
    #         user = int(user)
    #         while True:
    #             neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
    #             if neg_item not in train_user_set[user]:
    #                 break
    #         neg_items.append(neg_item)
    #     return neg_items
    #
    # feed_dict = {}
    # entity_pairs = train_entity_pairs[start:end].to(device)
    # feed_dict['users'] = entity_pairs[:, 0]
    # feed_dict['pos_items'] = entity_pairs[:, 1]
    # feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
    #                                                             train_user_set)).to(device)
'''
_get_topk_feed_data 和 _get_user_record 函数：

    _get_topk_feed_data：将特定用户与若干物品配对，生成模型评估时的输入格式。
    _get_user_record：将训练或测试数据处理成用户历史交互记录的字典，用于 Top-K 评估。

'''
def _show_recall_info(recall_zip):
    res = ""
    for i, j in recall_zip:
        res += "K@%d:%.4f  "%(i,j)
    logging.info(res)

def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item, 0])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
'''
ctr_eval 函数通过 ROC-AUC 和 F1-Score 指标来评估模型的点击预测能力。

'''
def ctr_eval(model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:

        batch = get_feed_dict(data, start, start + args.batch_size)
        labels = data[start:start + args.batch_size, 2]
        _, scores, _, _ = model(batch)
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1
'''
topk_eval 函数计算推荐系统的召回率。对于每个用户，模型会生成对未交互物品的评分，并计算 Top-K 推荐列表中的命中率。
'''
def topk_eval(model, train_data, data):
    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    item_set = set(train_data[:, 1].tolist() + data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)
            _, scores, _, _ = model(batch)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()

    recall = [np.mean(recall_list[k]) for k in k_list]
    return recall
    # _show_recall_info(zip(k_list, recall))

#####transGNN
g_trans_u_embeddings=[]
g_trans_i_embeddings=[]


if __name__ == '__main__':

    #####transGNN
    #global g_trans_u_embeddings, g_trans_i_embeddings
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setproctitle.setproctitle('proc_title')
    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load TransGNN Data')

    coach = Coach(handler)

    coach.run()

    #####end


    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    # train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_cf], np.int32))
    # eval_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in eval_cf], np.int32))
    # LongTensor
    # labels = torch.FloatTensor(np.array(cf[2] for cf in train_cf))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False
    '''
    加载数据和模型：

    数据通过 load_data 函数加载，并分为训练集和测试集。
    初始化模型对象 Recommender 和优化器 Adam。

训练：

    每轮训练对训练数据进行随机打乱（shuffle）。
    按批次从训练集中提取数据，通过 get_feed_dict 生成模型的输入字典。
    计算损失，进行反向传播更新模型参数。

评估：

    定期调用 ctr_eval 测试模型的性能指标。
    使用 PrettyTable 打印训练和测试结果。
    
    '''

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf = train_cf[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf, s, s + args.batch_size)
            batch_loss, _, _, _ = model(batch)
            # batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            # cor_loss += batch_cor
            s += args.batch_size

        train_e_t = time()
        # tsne_plot(model.all_embed, epoch)
        # if epoch % 10 == 9 or epoch == 1:
        if 1:
            """testing"""
            test_s_t = time()
            # ret = test(model, user_dict, n_params)
            test_auc, test_f1 = ctr_eval(model, test_cf_pairs)

            test_e_t = time()
            # ctr_info = 'epoch %.2d  test auc: %.4f f1: %.4f'
            # logging.info(ctr_info, epoch, test_auc, test_f1)
            train_res = PrettyTable()
            # train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision",
            #                          "hit_ratio", "auc"]
            # train_res.add_row(
            #     [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'],
            #      ret['precision'], ret['hit_ratio'], ret['auc']]
            # )
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "test auc", "test f1"]
            train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_auc, test_f1]
                )
            # train_res.field_names = ["Recall@5", "Recall@10", "Recall@20", "Recall@50", "Recall@100"]
            # train_res.add_row([recall[0], recall[1], recall[2], recall[3], recall[4]])
            print(train_res)
    print('early stopping at %d, test_auc:%.4f' % (epoch-30, cur_best_pre_0))
