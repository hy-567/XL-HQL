import argparse, os
from xlhql.utils.utils_hql import *
from train import construct_hyper_param, get_models
from sklearn.model_selection import KFold

#show
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

import gensim

##获得各个位置的混合矩阵以及F1_score
def get_agg_evaluate(y_true,y_pred):
    # 1.计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(cm, index=['None', 'COUNT', 'AVG', 'MIN', 'MAX', 'SUM'], columns=['None', 'COUNT', 'AVG', 'MIN', 'MAX', 'SUM'])  # 数据有5个类别
    # 画出混淆矩阵
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues",fmt='d')
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('agg_confusion.png', bbox_inches='tight')
    plt.show()

    f1=f1_score(y_true, y_pred, average='micro')
    print('Micro f1-score', f1)
    return f1


def get_cond_evaluate(y_true, y_pred):
    # 1.计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(cm, index=['0', '1', '2', '3', '4'], columns=['0', '1', '2', '3', '4'])  # 数据有5个类别
    # 画出混淆矩阵
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues",fmt='d')
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('cond_confusion.png', bbox_inches='tight')
    plt.show()

    f1 = f1_score(y_true, y_pred, average='micro')
    print('Micro f1-score', f1)
    return f1

def get_sel_evaluate(y_true, y_pred):
    # 1.计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(cm, index=['0','1'], columns=['0','1'])  # 数据有2个类别
    # 画出混淆矩阵
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues",fmt='d')
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('sel_confusion.png', bbox_inches='tight')
    plt.show()

    f1 = f1_score(y_true, y_pred, average='micro')
    print('Micro f1-score', f1)
    return f1

def get_op_evaluate(y_true, y_pred):
    # 1.计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(cm, index=['UNKNOWN', '=', '!=' , 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN','LIKE'], columns=['UNKNOWN', '=', '!=' , 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN','LIKE'])  # 数据有5个类别
    # 画出混淆矩阵
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 12}, cmap="Blues",fmt='d')
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('op_confusion.png', bbox_inches='tight')
    plt.show()

    f1 = f1_score(y_true, y_pred, average='micro')
    print('Micro f1-score', f1)
    return f1

def gen_querys(sql_i,hds,cs):
    res=[]
    agg_ops = ['NONE', 'COUNT', 'AVG', 'MIN', 'MAX', 'SUM']
    cond_ops = ['UNKNOWN', '=', '!=', 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN', 'LIKE']
    for b, sql_i1 in enumerate(sql_i):
        pr_spro=hds[b][sql_i1['sel']]
        pr_agg=agg_ops[sql_i1['agg']]
        if pr_agg !='NONE':
            s="select ("+pr_spro+")"
        else:
            s="select "+pr_spro
        s+=" from "+cs[b]
        if len(sql_i1['conds'])>0:
            s += " where "
            count=0
            for ss in sql_i1['conds']:
                count+=1
                s+= cs[b]+'.'+hds[b][ss[0]]
                if ss[1]!=0:
                    s+=' '+cond_ops[ss[1]]+' :!value '
                if count!=len(sql_i1['conds']):
                    s+='and'
        res.append(s)
    return res

def predict(test_loader, model,st_pos):
    model.eval()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_lx = 0  # of logical form acc
    cnt_sel = 0
    cnt_where = 0

    results = []
    count_sel_pr = []
    count_sel_gt = []

    count_agg_pr = []
    count_agg_gt = []
    count_cond_pr = []
    count_cond_gt = []
    count_op_pr = []
    count_op_gt = []

    for iB, t in enumerate(test_loader):
        cnt += len(t)  ##batch_size

        if cnt < st_pos:
            continue
        # Get fields  获得token
        nlu, nlu_t, sql_i, hds, hs_token,cs = get_fields(t)
        g_sc, g_sa, g_wn, g_wc, g_wo = get_g(sql_i)

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo = model(nlu, nlu_t, hds, hs_token, g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc)

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo)

        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, nlu)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
            cnt_wc1_list, cnt_wo1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, pr_sc, pr_sa, pr_wn, pr_wc,
                                                         pr_wo,
                                                         sql_i, pr_sql_i, mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list)

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_lx += sum(cnt_lx1_list)

        for i,j in zip(cnt_sc1_list,cnt_sa1_list):
            if i==1 and j==1:
                cnt_sel+=1

        for x,y,z in zip(cnt_wn1_list,cnt_wc1_list,cnt_wo1_list):
            if x==y==z==1:
                cnt_where+=1

        count_agg_pr.extend(pr_sa)
        count_agg_gt.extend(g_sa)

        count_cond_pr.extend(pr_wn)
        count_cond_gt.extend(g_wn)
        #
        # ##pr_sql_q = gen_querys(pr_sql_i,hds,cs)
        for g,t in zip(g_wo,pr_wo):
            if len(g)==len(t) and len(g)>=1:
                count_op_pr.extend(t)
                count_op_gt.extend(g)
        #
        for i,j in zip(g_sc,pr_sc):
            if i==0:
                count_sel_gt.append(0)
            else:
                count_sel_gt.append(1)

            if j==0:
                count_sel_pr.append(0)
            else:
                count_sel_pr.append(1)

        # for i,es in enumerate(cnt_lx1_list):
        #     if es==1:
        #         try:
        #             if is_true(t[i]['hql-real'], pr_sql_q[i]):
        #                 print("True")
        #                 print(t[i]['hql-real'])
        #                 print(pr_sql_q[i])
        #                 print('******')
        #                 test_acc += 1
        #         except IndexError:
        #             print('错误--')
        #             print(t[i]['hql-real'])
        #
        #     else:
        #         pass
        #         print("False")
        #         print(t[i]['hql-real'])
        #         print(pr_sql_q[i])

        # print(123)
        # for b, (real_sql_i1, pr_sql_q1) in enumerate(zip(t, pr_sql_q)):
        #     real_hql = real_sql_i1['hql-real']
        #     pre_hql = pr_sql_q1

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_lx = cnt_lx / cnt
    acc_sel = cnt_sel / cnt
    acc_where = cnt_where / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_lx, acc_sel, acc_where]
    print(acc)
    # f_agg=get_agg_evaluate(count_agg_gt,count_agg_pr)
    # print("F1-agg: {:0.5f}",f_agg)
    # f_wn=get_cond_evaluate(count_cond_gt,count_cond_pr)
    # print("F1-Cond: {:0.5f}", f_wn)
    # f_op=get_op_evaluate(count_op_gt,count_op_pr)
    # print("F1-OP: {:0.5f}", f_op)
    # f_sel = get_sel_evaluate(count_sel_gt, count_sel_pr)
    # print("F1-SEL: {:0.5f}", f_sel)
    return acc

## Set up hyper parameters and paths
parser = argparse.ArgumentParser()
parser.add_argument("--trained", default=False, action='store_true')
parser.add_argument("--bS", default=32, type=int, help="Batch size")
parser.add_argument("--accumulate_gradients", default=1, type=int)
parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")
parser.add_argument("--model_type", default='xlnet', type=str,
                        help="Type of model.")
parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--hS", default=32, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")
parser.add_argument('--project', default=False, action="store_true")
parser.add_argument('--type', default=False, action='store_true')
parser.add_argument('--rule', default=False, action='store_true')

args = parser.parse_args()
path_wikisql = './data_and_model'

if args.project:
    #Mixed
    HQL_data=data_merge(path_wikisql,False, 12)
    hql_np = np.array(HQL_data)
else:
    project_data = data_merge(path_wikisql, False, 12)
    hql_project = [{key: value} for key, value in project_data.items()]
    project_data = np.array(hql_project)
##交叉验证
skf = KFold(n_splits=10, shuffle=True, random_state=42)

word_model = gensim.models.Word2Vec.load('data_and_model/hql_word2vec.model')
word_emb = word_model.wv
del word_model
model=get_models(args,word_emb=word_emb)

def get_real_dataset(datasets):
    res = []
    for datas in datasets:
        for data in datas.values():
            res.extend(data)
    return res

# test_loader = torch.utils.data.DataLoader(
#     batch_size=32,
#     dataset=hql_np,
#     shuffle=True,
#     num_workers=0,
#     collate_fn=lambda x: x  # now dictionary values are not merged!
# )


# results = predict(test_loader,
#                        model,
#                        st_pos=0,
#                        )


# Run prediction
acc_temp=[]
with torch.no_grad():
    if args.project:
        for train_index, test_index in skf.split(hql_project):
            train_data, test_data = project_data[train_index].tolist(), project_data[test_index].tolist()
            train_data = get_real_dataset(train_data)
            test_data = get_real_dataset(test_data)
            train_loader, test_loader = get_loader_wikisql(train_data, test_data, args.bS, shuffle_train=True,
                                                           shuffle_test=True)
            results = predict(test_loader,
                           model,
                           st_pos=0,
                           )
            acc_temp.append(results[6])
        print(np.mean(acc_temp))
    else:
        for train_index, test_index in skf.split(HQL_data):
            train_data, test_data = hql_np[train_index].tolist(), hql_np[test_index].tolist()
            train_loader, test_loader = get_loader_wikisql(train_data, test_data, args.bS, shuffle_train=True,
                                                       shuffle_test=True)
            results = predict(test_loader,
                       model,
                       st_pos=0,
                       )
            acc_temp.append(results[6])
        print(np.mean(acc_temp))

