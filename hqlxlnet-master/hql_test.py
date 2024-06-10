import argparse, os
import re

from sqlova.utils.utils_hql import *
from train import construct_hyper_param, get_models
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
from sklearn.model_selection import KFold

def gen_querys(sql_i,hds,cs):
    res=[]
    agg_ops = ['NONE', 'COUNT', 'AVG', 'MIN', 'MAX', 'SUM']
    cond_ops = ['UNKNOWN', '=', '!=', 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN', 'LIKE']
    for b, sql_i1 in enumerate(sql_i):
        pr_spro=hds[b][sql_i1['sel']]
        pr_agg=agg_ops[sql_i1['agg']]
        s = "SELECT "
        if pr_agg !='NONE':
            if sql_i1['sel']==0:
                s+=pr_agg+"("+pr_spro+")"
            else:
                s+=pr_agg+"("+cs[b]+"."+pr_spro+")"
        else:
            if sql_i1['sel']==0:
                s+=pr_spro
            else:
                s+=cs[b]+"."+pr_spro
        s+=" FROM "+cs[b]
        if len(sql_i1['conds'])>0:
            s += " WHERE "
            count=0
            for ss in sql_i1['conds']:
                count+=1
                s+= cs[b]+'.'+hds[b][ss[0]]
                if ss[1]!=0:
                    if ss[1]==3 or ss[1]==4:
                        s+=' '+cond_ops[ss[1]]
                    else:
                        s+=' '+cond_ops[ss[1]]+' :!value'
                if count!=len(sql_i1['conds']):
                    s+=' AND '
        res.append(s)
    return res

def get_conditions(hql_query):
    conditions = hql_query.split("WHERE", 1)[1].strip().split("AND")
    return [condition.strip() for condition in conditions]

def turn_big(hql_query):
    res =re.sub(r"select", "SELECT", re.sub(r"from", "FROM", re.sub(r"where", "WHERE", re.sub(r"and", "AND", hql_query))))
    res1 = re.sub(r"count","COUNT", re.sub(r"avg", "AVG", re.sub(r"min", "MIN", re.sub(r"max", "MAX", re.sub(r"sum", "SUM", res)))))
    return res1

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
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('cond_confusion.png', bbox_inches='tight')
    plt.show()

    f1 = f1_score(y_true, y_pred, average='micro')
    print('Micro f1-score', f1)
    return f1


def get_op_evaluate(y_true, y_pred):
    # 1.计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = pd.DataFrame(cm, index=['UNKNOWN', '=', '!=' , 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN','LIKE'], columns=['UNKNOWN', '=', '!=' , 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN','LIKE'])  # 数据有5个类别
    # 画出混淆矩阵
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues",fmt='d')
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('op_confusion.png', bbox_inches='tight')
    plt.show()

    f1 = f1_score(y_true, y_pred, average='micro')
    print('Micro f1-score', f1)
    return f1

def is_true(h1, h2):
    real = turn_big(h1)
    pre = turn_big(h2)

    if real == pre:
        return True
    elif 'SELECT' not in real:
        a=re.split(r"FROM", real)[-1]
        b=re.split(r"FROM", pre)[-1]
        if a == b:
            return True
    elif 'WHERE' in real:
        real_1 = get_conditions(real)
        pre_1 = get_conditions(pre)

        if set(sorted(real_1)) == set(sorted(pre_1)):
            return True
    else:
        return False

def predict(model_type,test_loader, model,model_bert,bert_config,tokenizer,st_pos,num_target_layers):
    model.eval()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_lx = 0  # of logical form acc
    cnt_sel=0
    cnt_where=0

    results = []

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
        nlu, nlu_t, sql_i, hds, cs = get_fields(t)
        g_sc, g_sa, g_wn, g_wc, g_wo = get_g(sql_i)

        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        ##放入bert中做词嵌入
        wemb_n, wemb_h, l_n, l_hpu, l_hs, nlu_tt = get_wemb_bert(model_type,bert_config, nlu_t, hds, model_bert, tokenizer,
                                                                 num_out_layers_n=num_target_layers)


        # score
        s_sc, s_sa, s_wn, s_wc, s_wo = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                             g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc)

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

        # count_agg_pr.extend(pr_sa)
        # count_agg_gt.extend(g_sa)
        #
        # count_cond_pr.extend(pr_wn)
        # count_cond_gt.extend(g_wn)

        # pr_sql_q = gen_querys(pr_sql_i,hds,cs)

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
    acc_sel = cnt_sel/cnt
    acc_where = cnt_where/cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_lx,acc_sel,acc_where]
    print(acc)

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
parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")
parser.add_argument("--num_target_layers",
                        default=1, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
parser.add_argument('--project', default=False, action="store_true")
parser.add_argument('--type', default=False, action='store_true')
parser.add_argument('--rule', default=False, action='store_true')
args = parser.parse_args()
path_wikisql = './data_and_model'


args.toy_model = False
args.toy_size = 12
skf = KFold(n_splits=10, shuffle=True, random_state=42)

if args.project:
    project_data = data_corss_merge(path_wikisql, args.toy_model, args.toy_size,type= args.type,rule=args.rule)
    hql_project = [{key: value} for key, value in project_data.items()]
    project_data = np.array(hql_project)
else:
    HQL_data=data_merge(path_wikisql,False, 12,type = args.type,rule=args.rule)
    hql_np = np.array(HQL_data)

if not args.trained:
    model, x_model, x_tokenizer, xl_config = get_models(args)
else:
    # To start from the pre-trained models, un-comment following lines.     ##加载已经训练好的模型
    if args.EG:
        path_model_bert = './model_xlnet_EG_best.pt'
        path_model = './model_EG_best.pt'
    else:
        path_model_bert = './cross_project/model_gpt2_best.pt'
        path_model = './cross_project/model_best.pt'

    model, x_model, x_tokenizer, xl_config = get_models(args, trained=True, path_model_bert=path_model_bert,
                                                        path_model=path_model)

def get_real_dataset(datasets):
    res = []
    for datas in datasets:
        for data in datas.values():
            res.extend(data)
    return res


acc_temp = []
with torch.no_grad():
    if args.project:
        for train_index, test_index in skf.split(hql_project):
            train_data, test_data = project_data[train_index].tolist(), project_data[test_index].tolist()
            test_data = get_real_dataset(test_data)
            _, test_loader = get_loader_wikisql(train_data, test_data, args.bS, shuffle_train=True,
                                                       shuffle_test=True)
            acc_dev = predict(args.model_type,test_loader, model=model, model_bert=x_model, bert_config=xl_config,
                   tokenizer=x_tokenizer,
                   st_pos=0,num_target_layers=args.num_target_layers,EG=args.EG,beam_size=args.beam_size)
            acc_temp.append(acc_dev[6])
        test_ave_acc = np.mean(acc_temp)
        print(test_ave_acc)
    else:
        for train_index, test_index in skf.split(HQL_data):
            train_data, test_data = hql_np[train_index].tolist(), hql_np[test_index].tolist()
            train_loader, test_loader = get_loader_wikisql(train_data, test_data, args.bS, shuffle_train=True,
                                                       shuffle_test=True)
            results = predict(args.model_type,test_loader, model=model, model_bert=x_model, bert_config=xl_config,
                   tokenizer=x_tokenizer,
                   st_pos=0,num_target_layers=args.num_target_layers)
            acc_temp.append(results[6])
        print(np.mean(acc_temp))