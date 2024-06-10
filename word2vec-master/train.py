# Copyright 2019-present NAVER Corp.
# Apache License v2.0

import os, sys, argparse, re, json

from sklearn.model_selection import KFold
from xlhql.model.hqlmodel.hql_models import *

#show
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

import gensim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=False, action='store_true')

    parser.add_argument("--trained", default=False, action='store_true')
    parser.add_argument("--project", default=False, action='store_true')
    parser.add_argument("--type", default=False, action='store_true')
    parser.add_argument("--rule", default=False, action='store_true')

    parser.add_argument('--tepoch', default=200, type=int)
    parser.add_argument("--bS", default=32, type=int,help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=32, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    args = parser.parse_args()

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.toy_model = False
    args.toy_size = 12  #设置bert最后一层的层数

    return args




def get_opt(model):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0)
    return opt


def get_models(args, word_emb=None):
    # some constants
    agg_ops = {'NONE', 'COUNT', 'AVG', 'MIN', 'MAX', 'SUM'}
    cond_ops = ['UNKNOWN', '=', '!=', 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN','LIKE']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")

    args.iS=96
    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = XL_HQL(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops,word_emb)
    model = model.to(device)


    if args.trained:
        if torch.cuda.is_available():
            res = torch.load('./w2vec_model.pt')
            #res = torch.load('cross/w2vec_model_cross.pt')
        else:
            res = torch.load('./w2vec_model.pt', map_location='cpu')

        model.load_state_dict(res['model'])
    return model

def train(train_loader, model, opt,  st_pos=0,  dset_name='train'):
    model.train()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_lx = 0  # of logical form acc

    for iB, t in enumerate(train_loader):
        cnt += len(t)   ##batch_size

        if cnt < st_pos:
            continue
        # Get fields  获得token
        nlu, nlu_t, sql_i, hds,hs_token,cs = get_fields(t)

        g_sc, g_sa, g_wn, g_wc, g_wo = get_g(sql_i)

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo = model(nlu,nlu_t, hds,hs_token,g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, g_sc, g_sa, g_wn, g_wc, g_wo)

        # Calculate gradient
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, nlu)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list= get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo,pr_sc, pr_sa, pr_wn, pr_wc, pr_wo,
                                                      sql_i, pr_sql_i,mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,cnt_wo1_list)

        # statistics
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_lx += sum(cnt_lx1_list)

        # count_agg_pr.extend(pr_sa)
        # count_agg_gt.extend(g_sa)
        #
        # count_cond_pr.extend(pr_wn)
        # count_cond_gt.extend(g_wn)

        # for g,t in zip(g_wo,pr_wo):
        #     if len(g)==len(t) and len(g)>=1:
        #         count_op_pr.extend(t)
        #         count_op_gt.extend(g)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_lx = cnt_lx / cnt


    # f_agg=get_agg_evaluate(count_agg_gt,count_agg_pr)
    # f_wn=get_cond_evaluate(count_cond_gt,count_cond_pr)
    # f_op=get_op_evaluate(count_op_gt,count_op_pr)

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_lx]

    aux_out = 1

    return acc, aux_out


def test(test_loader, model, st_pos=0):
    model.eval()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    #cnt_wv = 0  # of where-value
    cnt_lx = 0  # of logical form acc

    for iB, t in enumerate(test_loader):
        cnt += len(t)  ##batch_size

        if cnt < st_pos:
            continue
        # Get fields  获得token
        nlu, nlu_t, sql_i, hds,hs_token, cs = get_fields(t)

        g_sc, g_sa, g_wn, g_wc, g_wo = get_g(sql_i)

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo = model(nlu,nlu_t, hds,hs_token,g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, g_sc, g_sa, g_wn, g_wc, g_wo)

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo)

        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, nlu)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
            cnt_wc1_list, cnt_wo1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, pr_sc, pr_sa, pr_wn, pr_wc,
                                                         pr_wo,
                                                         sql_i, pr_sql_i, mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list)

        # statistics
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_lx += sum(cnt_lx1_list)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_lx = cnt_lx / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_lx]

    return acc

def get_real_dataset(datasets):
    res = []
    for datas in datasets:
        for data in datas.values():
            res.extend(data)
    return res


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_lx = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f},acc_lx: {acc_lx:.3f}"
    )


if __name__ == '__main__':
    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = './data_and_model"'
    path_wikisql = './data_and_model'
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'
    skf = KFold(n_splits=10, shuffle=True, random_state=42)

    ##merge_data
    if args.project:
        project_data = data_corss_merge(path_wikisql, args.toy_model, args.toy_size,rule= args.rule)
        hql_project = [{key: value} for key, value in project_data.items()]
        project_data = np.array(hql_project)
    else:
        HQL_data = data_merge(path_wikisql, args.toy_model, args.toy_size, rule=args.rule)
        hql_np = np.array(HQL_data)

    word_model = gensim.models.Word2Vec.load('data_and_model/hql_word2vec.model')
    word_emb = word_model.wv
    del word_model


    ## 4. Build & Load models
    model=get_models(args,word_emb=word_emb)

    ## 5. Get optimizers
    if args.do_train:
        opt = get_opt(model)
    ## 6. Train
        acc_lx_t_best = -1
        epoch_best = -1

        if args.project:
            for epoch in range(args.tepoch):
                print('Epoch %d @ %s' % (epoch + 1, datetime.datetime.now()))
                test_ave_acc = 0
                acc_temp = []
                for train_index, test_index in skf.split(hql_project):
                    train_data, test_data = project_data[train_index].tolist(), project_data[test_index].tolist()
                    train_data = get_real_dataset(train_data)
                    test_data = get_real_dataset(test_data)
                    train_loader, test_loader = get_loader_wikisql(train_data, test_data, args.bS, shuffle_train=True,
                                                                   shuffle_test=True)
                    acc_train, aux_out_train = train(train_loader,
                                             model,
                                             opt,
                                             st_pos=0,
                                             dset_name='train')

                    # check DEV
                    with torch.no_grad():
                        acc_dev = test(test_loader,
                                    model,
                                    st_pos=0,
                                   )
                        acc_temp.append(acc_dev[-1])

                    print_result(epoch + 1, acc_train, 'train')
                    print_result(epoch + 1, acc_dev, 'dev')

                test_ave_acc = np.mean(acc_temp)
                if test_ave_acc > acc_lx_t_best:
                    acc_lx_t_best = test_ave_acc
                    epoch_best = epoch + 1
                    state = {'model': model.state_dict()}
                    torch.save(state, os.path.join('./cross', 'w2vec_model_cross.pt'))

                print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
        else:
            for epoch in range(args.tepoch):
                print('Epoch %d @ %s' % (epoch + 1, datetime.datetime.now()))
                test_ave_acc = 0
                acc_temp = []
                for train_index, test_index in skf.split(HQL_data):
                    train_data, test_data = hql_np[train_index].tolist(), hql_np[test_index].tolist()
                    train_loader, test_loader = get_loader_wikisql(train_data, test_data, args.bS, shuffle_train=True,
                                                                   shuffle_test=True)
                    acc_train, aux_out_train = train(train_loader,
                                             model,
                                             opt,
                                             st_pos=0,
                                             dset_name='train')

                    # check DEV
                    with torch.no_grad():
                        acc_dev = test(test_loader,
                                    model,
                                    st_pos=0,
                                   )
                        acc_temp.append(acc_dev[-1])

                    print_result(epoch + 1, acc_train, 'train')
                    print_result(epoch + 1, acc_dev, 'dev')

                test_ave_acc = np.mean(acc_temp)

                if test_ave_acc > acc_lx_t_best:
                    acc_lx_t_best = test_ave_acc
                    epoch_best = epoch + 1
                    state = {'model': model.state_dict()}
                    torch.save(state, os.path.join('.', 'w2vec_model.pt'))

                print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
            