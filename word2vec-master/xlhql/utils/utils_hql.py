# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang

import os, json
import random as rd
import re
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from PyDeepLX import PyDeepLX

from .utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load data -----------------------------------------------------------------------------------------------



def contains_non_english(text):
    # 定义正则表达式，匹配非英文字符
    non_english_pattern = re.compile(r'[^\x00-\x7F]')
    # 搜索是否存在非英文字符
    if non_english_pattern.search(text):
        return True
    else:
        return False

def translate(ss):
    return PyDeepLX.translate(ss)

def get_question(d):
    s=[]
    if d['HQLMethodComment'] != '':
        s.append('The MethodComment is')
        #s.append(d['HQLMethodComment'].replace('\r','').replace('\n','').replace('\t','').replace(':','').lower())
        s.append(' '.join(re.split(r'_|\.| |,|;|\)|\(|\{|\}|\<|\>|:|\*|@', d['HQLMethodComment'], flags=0)))
    if d['HQLMethod'] != '':
        s.append('The MethodName is')
        s.append(d['HQLMethod'])
    if len(d['HQLMethodParName'])>0:
        s.append('The MethodParameterName is')
        s.append(' '.join(list(map(lambda x:x,d['HQLMethodParName']))))
    if len(d['calledInMethod'])>0:
        s.append('The contain of CallContextMethodName is')
        s.append(' '.join(list(map(lambda x:x,d['calledInMethod']))))
    count=0
    for t in d['calledInPar']:
        if len(t)==0:
            continue
        else:
            count+=1
            if count==1:
                s.append('The CallContextMethodParameterName is')
            s.append(t[0])
    res=' '.join(s).lower()
    return res

def get_question_token(identifiter):
    result = []
    for s in re.split(r':| |,|;|\)|\(|\{|\}|\<|\>|\*|@|#', identifiter, flags=0):
        result.append(s.lower())

    return result
##最大公共子串

def getNumofCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2+1)] for j in range(lstr1+1)] # 多一位
    maxNum = 0   # 最长匹配长度
    p = 0    # 匹配的起始位
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maxNum:
                    maxNum = record[i+1][j+1]
                    p = i + 1
    return str1[p-maxNum:p]

def get_hql(d,pro,c):
    conds={}
    hql_query=d['splitHql']
    hql_sel = hql_query.get('select')
    agg = {'NONE': 0, 'COUNT': 1, 'AVG': 2, 'MIN': 3, 'MAX': 4, 'SUM': 5}
    COND_OPS = {'UNKNOWN': 0, '=': 1, '!=': 2, 'IS NULL': 3, 'IS NOT NULL': 4, '<': 5, '>': 6, 'IN': 7, 'NOT IN': 8,'LIKE': 9}
    conds['agg']=agg.get(hql_sel.get('agg','None').upper())
    ##sel_pro在pro中的位置
    if hql_sel['column'] == c:
        conds['sel_pro']=0
    elif hql_sel['column'] in pro:
        conds['sel_pro']=pro.index(hql_sel['column'])
    elif hql_sel['column'].find('.') != -1:
        s = re.split(c + '.', hql_sel['column'], 1)[-1]
        if s in pro:
            conds['sel_pro']=pro.index(s)
        else:
            common_s = getNumofCommonSubstr(hql_sel['column'], c)
            conds['sel_pro']=pro.index(common_s)

    hql_where = hql_query.get('where')
    gt_value = []
    if len(hql_where) > 0:
        gt_value.append(len(hql_where))
        for dic_where in hql_where:
            temp = []
            where_op = dic_where.get('op', 'UNKNOWN')
            if where_op == '>=':
                where_op = '>'
            elif where_op == '<=':
                where_op = '<'
            elif where_op.upper() == 'NOT LIKE':
                where_op = 'LIKE'
                print('NOT LIKE被替换')
            temp.append(COND_OPS.get(where_op.upper(), 0))

            where_pro = dic_where.get('column')
            if where_pro in pro:
                temp.append(pro.index(where_pro))
            else:
                if where_pro.find('.')!=-1:
                    s=re.split(c+'.',where_pro,1)[-1]
                    try:
                        temp.append(pro.index(s))
                    except:
                        temp.append(pro.index(where_pro.split('.')[-1]))
            gt_value.append(temp)
    else:
        gt_value.append(0)
        temp = []
        where_pro = 'None'
        temp.append('None')
        temp.append(where_pro)
        gt_value.extend(temp)
    conds['where']=gt_value
    return conds

def get_header_token(data):
    res=[]
    for d in data:
        res.append(snake_case_split(camel_case_split(d)))
    return res

def get_question_type(d):
    s=[]
    if d['HQLMethodComment'] != '':
        s.append('The MethodComment is')
        s.append(d['HQLMethodComment'].replace('\r','').replace('\n','').replace('\t',''))

    s.append('The MethodSignature is')
    s.extend(get_method_question(d))

    s.append('The contain of CallContextMethodSignature are')
    s.extend(get_CallMethod_question(d))

    res=' '.join(s)
    return res

##拼接构造方法签名
def get_method_question(d):
    s = []
    s.append(d['HQLMethod'])
    s.append("(")
    flag = False
    if len(d['HQLMethodParName'])>0:
        for i,j in zip(d['HQLMethodParType'],d['HQLMethodParName']):
            s.append(i + " " + j)
            s.append(",")
        flag = True
    if flag:
        s[-1] = ")"
    else:
        s.append(")")
    return s

##拼接构造call-context签名
def get_CallMethod_question(d):
    s = []
    flag = True
    if len(d['calledInMethod']) > 0:
        for i,j,k in zip(d['calledInMethod'],d['calledInParType'],d['calledInPar']):
            s.append(i+"( ")
            for jj,kk in zip(j,k):
                if(jj!=None):
                    s.append(jj + "" + kk)
                    s.append(",")
            s.append(" )")
        flag = False

    if flag:
        s.append("NULL")
    return s

# def get_question_token(identifiter):
#     result = []
#     for word in re.split(r':| |,|;|\{|\}|\<|\>|\*|@|#', identifiter, flags=0):
#         if word:
#             result.append(word)
#
#     return result

def data_merge(path_hql,toy_model,toy_size,rule):
    path_sql = os.path.join(path_hql, 'dataset_hql.json')
    DATA=[]
    with open(path_sql) as inf:
        for idx, line in enumerate(inf):
            if toy_model and idx >= toy_size:
                break
            hql_info_list = json.loads(line.strip())
            DATA.append(hql_info_list)

    hql_data=[]
    count=0

    for i,d in enumerate(DATA):
        temp={}
        if 'GROUP BY' in d['cleanedHql'] or "ORDER BY" in d['cleanedHql']:
            continue
        # if contains_non_english(d['HQLMethodComment']):
        #     count+=1
        #     d['HQLMethodComment'] = translate(d['HQLMethodComment'])

        temp['question']=get_question(d)
        temp['question_token']=get_question_token(temp['question'])

        gt_class = d['splitHql']['from']
        c_t= re.split("\.", gt_class)
        if len(c_t) % 2 == 0:
            a1 = '.'.join(c_t[x] for x in range(int((len(c_t) + 1) / 2)))
            a2 = '.'.join(c_t[x] for x in range(int((len(c_t) + 1) / 2), len(c_t)))
            if a1 == a2:
                gt_class = a1

        temp['class'] = gt_class
        t = d['column']
        t.insert(0, gt_class)
        if not rule:
            temp['header'], fs = all_header(t)
        else:
            temp['header'], fs = clean_hader(t, gt_class)

        temp['hql'] = get_hql(d,temp['header'],temp['class'])

        if temp['hql']['where'][0]>4:
            continue

        temp['header_token']=get_header_token(temp['header'])
        hql_data.append(temp)

    print(count)
    return hql_data

def data_corss_merge(path_hql,toy_model,toy_size,rule):
    path_sql = os.path.join(path_hql, 'dataset_hql.json')
    DATA=[]
    with open(path_sql) as inf:
        for idx, line in enumerate(inf):
            if toy_model and idx >= toy_size:
                break
            hql_info_list = json.loads(line.strip())
            DATA.append(hql_info_list)

    count=0
    res_map = dict()
    for i,d in enumerate(DATA):
        temp={}
        if 'GROUP BY' in d['cleanedHql'] or "ORDER BY" in d['cleanedHql']:
            continue

        temp['question']=get_question(d)

        temp['question_token']=get_question_token(temp['question'])
        gt_class = d['splitHql']['from']
        c_t= re.split("\.", gt_class)
        if len(c_t) % 2 == 0:
            a1 = '.'.join(c_t[x] for x in range(int((len(c_t) + 1) / 2)))
            a2 = '.'.join(c_t[x] for x in range(int((len(c_t) + 1) / 2), len(c_t)))
            if a1 == a2:
                gt_class = a1

        temp['class'] = gt_class
        t = d['column']
        t.insert(0, gt_class)

        if not rule:
            temp['header'], fs = all_header(t)
        else:
            temp['header'], fs = clean_hader(t, gt_class)

        temp['hql'] = get_hql(d,temp['header'],temp['class'])

        if temp['hql']['where'][0]>4:
            continue

        if d["projectName"] in res_map:
            res_map[d["projectName"]].append(temp)
        else:
            res_map.setdefault(d["projectName"], []).append(temp)

    print(count)
    return res_map

def load_wikisql(path_wikisql, toy_model, toy_size, bert=False, no_w2i=False, no_hs_tok=False, aug=False):
    # Get data
    train_data, train_table = load_wikisql_data(path_wikisql, mode='train', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok, aug=aug)
    # Get word vector
    if no_w2i:
        w2i, wemb = None, None
    else:
        w2i, wemb = load_w2i_wemb(path_wikisql, bert)


    return train_data, train_table, w2i, wemb


def load_wikisql_data(path_wikisql, mode='train', toy_model=False, toy_size=10, no_hs_tok=False, aug=False):
    """ Load training sets
    """
    ##增强数据已加载用于数据增强的
    if aug:
        mode = f"aug.{mode}"
        print('Augmented data is loaded!')
    ##读取制定路径的Json文件
    path_sql = os.path.join(path_wikisql, mode+'_tok.jsonl')
    if no_hs_tok:
        path_table = os.path.join(path_wikisql, mode + '.tables.jsonl')
    else:
        path_table = os.path.join(path_wikisql, mode+'_tok.tables.jsonl')

    data = []
    table = {}
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break

            t1 = json.loads(line.strip())
            data.append(t1)

    with open(path_table) as f:
        for idx, line in enumerate(f):
            if toy_model and idx > toy_size:
                break

            t1 = json.loads(line.strip())
            table[t1['id']] = t1

    return data, table

def all_header(hs):
    res = []
    flag, count = True, 0
    for hs1 in hs:
        if len(hs1) > 50:
            count += 1
        if hs1 in res:
            continue
        else:
            res.append(hs1)
    if count > 10:
        flag = False

    return res, flag

def load_w2i_wemb(path_wikisql, bert=False):
    """ Load pre-made subset of TAPI.
    """
    if bert:
        with open(os.path.join(path_wikisql, 'w2i_bert.json'), 'r') as f_w2i:
            w2i = json.load(f_w2i)
        wemb = load(os.path.join(path_wikisql, 'wemb_bert.npy'), )
    else:
        with open(os.path.join(path_wikisql, 'w2i.json'), 'r') as f_w2i:
            w2i = json.load(f_w2i)

        wemb = load(os.path.join(path_wikisql, 'wemb.npy'), )
    return w2i, wemb

def get_loader_wikisql(data_train, test_dev, bS, shuffle_train=True, shuffle_test=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=0,
        collate_fn=lambda x:x # now dictionary values are not merged!
    )

    ##torch.utils.data.DataLoader 主要是对数据进行 batch 的划分..在训练模型时使用到此函数，用来 把训练数据分成多个小组 ，此函数 每次抛出一组数据 。直至把所有的数据都抛出。就是做一个数据的初始化。
    test_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=test_dev,
        shuffle=shuffle_test,
        num_workers=0,
        collate_fn=lambda x:x # now dictionary values are not merged!
    )

    return train_loader, test_loader

##当是类的时候就为该类。若是类. 则为属性 否则直接添加
def clean_hader(hs,c):
    res=[]
    flag,count=True,0  ##判断一个字符是否超过50个大小
    for hs1 in hs:
        if hs1==c:
            if len(hs1)>50:
                count+=1
            res.append(hs1)

        elif hs1.find('.')!=-1:
            if hs1.startswith(c+'.'):
                s = re.split(c + '.', hs1, 1)[-1]
                if s in res:
                    continue
                else:
                    if len(s)>50:
                        count+=1
                    res.append(s)
            else:
                if len(hs1)>50:
                    count+=1
                res.append(hs1)
        else:
            if len(hs1) > 50:
                count += 1
            res.append(hs1)
    if count>10:
        flag=False
    return res,flag

def get_fields(t1s):

    nlu, nlu_t, sql_i, hs,hs_token,cs = [], [], [], [],[],[]
    for t1 in t1s:
        nlu1=t1['question']
        nlu_t1=t1['question_token']
        sql_i1=t1['hql']
        hs1=t1['header']
        hs1_token=t1['header_token']
        g_cs=t1['class']
        ##header的清洗 由于类.pro的性质 并且由于只关注的任务是预测最后一个点后面的属性

        nlu.append(nlu1)
        nlu_t.append(nlu_t1)
        sql_i.append(sql_i1)
        hs.append(hs1)
        hs_token.append(hs1_token)
        cs.append(g_cs)

    return nlu, nlu_t, sql_i, hs,hs_token,cs


# Embedding -------------------------------------------------------------------------

def word_to_idx1(words1, w2i, no_BE):
    w2i_l1 = []
    l1 = len(words1)  # +2 because of <BEG>, <END>


    for w in words1:
        idx = w2i.get(w, 0)
        w2i_l1.append(idx)

    if not no_BE:
        l1 += 2
        w2i_l1 = [1] + w2i_l1 + [2]

    return w2i_l1, l1


def words_to_idx(words, w2i, no_BE=False):
    """
    Input: [ ['I', 'am', 'hero'],
             ['You', 'are 'geneus'] ]
    output:

    w2i =  [ B x max_seq_len, 1]
    wemb = [B x max_seq_len, dim]

    - Zero-padded when word is not available (teated as <UNK>)
    """
    bS = len(words)
    l = torch.zeros(bS, dtype=torch.long).to(device) # length of the seq. of words.
    w2i_l_list = [] # shall be replaced to arr

    #     wemb_NLq_batch = []

    for i, words1 in enumerate(words):

        w2i_l1, l1 = word_to_idx1(words1, w2i, no_BE)
        w2i_l_list.append(w2i_l1)
        l[i] = l1

    # Prepare tensor of wemb
    # overwrite w2i_l
    w2i_l = torch.zeros([bS, int(max(l))], dtype=torch.long).to(device)
    for b in range(bS):
        w2i_l[b, :l[b]] = torch.LongTensor(w2i_l_list[b]).to(device)

    return w2i_l, l

def hs_to_idx(hs_t, w2i, no_BE=False):
    """ Zero-padded when word is not available (teated as <UNK>)
    Treat each "header tokens" as if they are NL-utterance tokens.
    """

    bS = len(hs_t)  # now, B = B_NLq
    hpu_t = [] # header pseudo-utterance
    l_hs = []
    for hs_t1 in hs_t:
        hpu_t  += hs_t1
        l_hs1 = len(hs_t1)
        l_hs.append(l_hs1)

    w2i_hpu, l_hpu = words_to_idx(hpu_t, w2i, no_BE=no_BE)
    return w2i_hpu, l_hpu, l_hs


# Encoding ---------------------------------------------------------------------

def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape


    # sort before packking
    l = array(l)
    perm_idx = argsort(-l)   ##从小到大进行排序
    perm_idx_inv = generate_perm_inv(perm_idx)

    # pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)

    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    # ipdb.set_trace()
    packed_wemb_l = packed_wemb_l.float() # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if last_only:
        # Take only final outputs for each columns.
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]

    wenc = wenc[perm_idx_inv]



    if return_hidden:
        # hout.shape = [number_of_directoin * num_of_layer, seq_len(=batch size), dim * number_of_direction ] w/ batch_first.. w/o batch_first? I need to see.
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)  # Is this correct operation?

        return wenc, hout, cout
    else:
        return wenc


def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs):
    wenc_hpu, hout, cout = encode( lstm,
                                   wemb_hpu,
                                   l_hpu,
                                   return_hidden=True,
                                   hc0=None,
                                   last_only=True )

    wenc_hpu = wenc_hpu.squeeze(1)
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)

    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    st = 0
    for i, l_hs1 in enumerate(l_hs):
        wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
        st += l_hs1

    return wenc_hs


# Statistics -------------------------------------------------------------------------------------------------------------------



def get_wc1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wc1 = []
    for i in range(conds[0]):
        wc1.append(conds[i + 1][1])
    return wc1


def get_wo1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wo1 = []
    for i in range(conds[0]):
        wo1.append(conds[i+1][0])
    return wo1

def get_g(sql_i):
    """ for backward compatibility, separated with get_g"""
    g_sc = []
    g_sa = []
    g_wn = []
    g_wc = []
    g_wo = []
    for b, psql_i1 in enumerate(sql_i):
        g_sc.append( psql_i1["sel_pro"] )   ##sel的目标列
        g_sa.append( psql_i1["agg"])    ##sel的AGG
        conds = psql_i1['where']

        if not psql_i1["agg"] < 0:
            g_wn.append( conds[0] )    ##conds的数量
            g_wc.append( get_wc1(conds) )  ##目标列索引
            g_wo.append( get_wo1(conds) )  ##操作符
        else:
            raise EnvironmentError
    return g_sc, g_sa, g_wn, g_wc, g_wo

def get_g_wvi_corenlp(t):
    g_wvi_corenlp = []
    for t1 in t:
        g_wvi_corenlp.append( t1['wvi_corenlp'] )
    return g_wvi_corenlp


def update_w2i_wemb(word, wv, idx_w2i, n_total, w2i, wemb):
    """ Follow same approach from SQLNet author's code.
        Used inside of generaet_w2i_wemb.
    """

    # global idx_w2i, w2i, wemb  # idx, word2vec, word to idx dictionary, list of embedding vec, n_total: total number of words
    if (word in wv) and (word not in w2i):
        idx_w2i += 1
        w2i[word] = idx_w2i
        wemb.append(wv[word])
    n_total += 1
    return idx_w2i, n_total

def make_w2i_wemb(args, path_save_w2i_wemb, wv, data_train, data_dev, data_test, table_train, table_dev, table_test):

    w2i = {'<UNK>': 0, '<BEG>': 1, '<END>': 2}  # to use it when embeds NL query.
    idx_w2i = 2
    n_total = 3

    wemb = [np.zeros(300, dtype=np.float32) for _ in range(3)]  # 128 is of TAPI vector.
    idx_w2i, n_total = generate_w2i_wemb(data_train, wv, idx_w2i, n_total, w2i, wemb)
    idx_w2i, n_total = generate_w2i_wemb_table(table_train, wv, idx_w2i, n_total, w2i, wemb)

    idx_w2i, n_total = generate_w2i_wemb(data_dev, wv, idx_w2i, n_total, w2i, wemb)
    idx_w2i, n_total = generate_w2i_wemb_table(table_dev, wv, idx_w2i, n_total, w2i, wemb)

    idx_w2i, n_total = generate_w2i_wemb(data_test, wv, idx_w2i, n_total, w2i, wemb)
    idx_w2i, n_total = generate_w2i_wemb_table(table_test, wv, idx_w2i, n_total, w2i, wemb)

    path_w2i = os.path.join(path_save_w2i_wemb, 'w2i.json')
    path_wemb = os.path.join(path_save_w2i_wemb, 'wemb.npy')

    wemb = np.stack(wemb, axis=0)

    with open(path_w2i, 'w') as f_w2i:
        json.dump(w2i, f_w2i)

    np.save(path_wemb, wemb)

    return w2i, wemb

def generate_w2i_wemb_table(tables, wv, idx_w2i, n_total, w2i, wemb):
    """ Generate subset of GloVe
        update_w2i_wemb. It uses wv, w2i, wemb, idx_w2i as global variables.

        To do
        1. What should we do with the numeric?
    """
    # word_set from NL query
    for table_id, table_contents in tables.items():

        # NLq = t1['question']
        # word_tokens = NLq.rstrip().replace('?', '').split(' ')
        headers = table_contents['header_tok'] # [ ['state/terriotry'], ['current', 'slogan'], [],
        for header_tokens in headers:
            for token in header_tokens:
                idx_w2i, n_total = update_w2i_wemb(token, wv, idx_w2i, n_total, w2i, wemb)
                # WikiSQL generaets unbelivable query... using state/territory in the NLq. Unnatural.. but as is
                # when there is slash, unlike original SQLNet which treats them as single token, we use
                # both tokens. e.g. 'state/terriotry' -> 'state'
                # token_spl = token.split('/')
                # for token_spl1 in token_spl:
                #         idx_w2i, n_total = update_w2i_wemb(token_spl1, wv, idx_w2i, n_total, w2i, wemb)

    return idx_w2i, n_total
def generate_w2i_wemb(train_data, wv, idx_w2i, n_total, w2i, wemb):
    """ Generate subset of GloVe
        update_w2i_wemb. It uses wv, w2i, wemb, idx_w2i as global variables.

        To do
        1. What should we do with the numeric?
    """
    # word_set from NL query
    for i, t1 in enumerate(train_data):
        # NLq = t1['question']
        # word_tokens = NLq.rstrip().replace('?', '').split(' ')
        word_tokens = t1['question_tok']
        # Currently, TAPI does not use "?". So, it is removed.
        for word in word_tokens:
            idx_w2i, n_total = update_w2i_wemb(word, wv, idx_w2i, n_total, w2i, wemb)
            n_total += 1

    return idx_w2i, n_total

def generate_w2i_wemb_e2k_headers(e2k_dicts, wv, idx_w2i, n_total, w2i, wemb):
    """ Generate subset of TAPI from english-to-korean dict of table headers etc..
        update_w2i_wemb. It uses wv, w2i, wemb, idx_w2i as global variables.

        To do
        1. What should we do with the numeric?
           Current version do not treat them specially. But this would be modified later so that we can use tags.
    """
    # word_set from NL query
    for table_name, e2k_dict in e2k_dicts.items():
        word_tokens_list = list(e2k_dict.values())
        # Currently, TAPI does not use "?". So, it is removed.
        for word_tokens in word_tokens_list:
            for word in word_tokens:
                idx_w2i, n_total = update_w2i_wemb(word, wv, idx_w2i, n_total, w2i, wemb)
                n_total += 1

    return idx_w2i, n_total


# BERT =================================================================================================================
def tokenize_nlu1(tokenizer, nlu1):
    nlu1_tok = tokenizer.tokenize(nlu1)
    return nlu1_tok


def tokenize_hds1(tokenizer, hds1):
    hds_all_tok = []
    for hds11 in hds1:
        sub_tok = tokenizer.tokenize(hds11)
        hds_all_tok.append(sub_tok)

def generate_inputs(tokenizer, nlu1_tok, hds1,max_length):
    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    for token in nlu1_tok:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)

    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)

    ##裁剪法
    if len(tokens)>max_length:
        tokens=tokens[:max_length]
        segment_ids=segment_ids[:max_length]
        if i_nlu[1]>max_length:
            i_nlu=(i_st_nlu,max_length)
        i1_hd=[]
        for i_s,i_e in i_hds:
            if i_e<max_length:
                i1_hd.append((i_s,i_e))
            elif i_s<max_length:
                i1_hd.append((i_s,max_length))
                break

        i_hds=i1_hd

    return tokens, segment_ids, i_nlu, i_hds

##将所有batch中的header_token的长度获取
def gen_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i_hds1 in i_hds:
        for i_hds11 in i_hds1:
            l_hpu.append(i_hds11[1] - i_hds11[0])

    return l_hpu

def get_bert_output_s2s(model_bert, tokenizer, nlu_t, hds, sql_vocab, max_seq_length):
    """
    s2s version. Treat SQL-tokens as pseudo-headers
    sql_vocab = ("sql select", "sql where", "sql and", "sql equal", "sql greater than", "sql less than")

    e.g.)
    Q: What is the name of the player with score greater than 15?
    H: Name of the player, score
    Input: [CLS], what, is, ...,
    [SEP], name, of, the, player, [SEP], score,
    [SEP] sql, select, [SEP], sql, where, [SEP], sql, and, [SEP], ...

    Here, input is tokenized further by WordPiece (WP) tokenizer and fed into BERT.

    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """


    l_n = []
    l_hs = []  # The length of columns for each batch
    l_input = []
    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []
    i_sql_vocab = []

    doc_tokens = []
    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        l_hs.append(len(hds1))


        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        nlu_tt.append(nlu_tt1)
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)

        l_n.append(len(nlu_tt1))
        #         hds1_all_tok = tokenize_hds1(tokenizer, hds1)



        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        # Combine hds1 and sql_vocab
        tokens1, segment_ids1, i_sql_vocab1, i_nlu1, i_hds1 = generate_inputs_s2s(tokenizer, nlu_tt1, hds1, sql_vocab)

        # i_hds1
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        l_input.append( len(input_ids1) )
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)
        i_sql_vocab.append(i_sql_vocab1)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, i_sql_vocab, \
           l_n, l_hpu, l_hs, l_input, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx

##
def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.
    传统词表示方法没有办法很好地处理未知或罕见的词汇。 WordPiece 和 BPE 是两种子词切分算法，两者非常相似。
    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []    ##总Mask

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    nlu_tt = []

    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        l_hs.append(len(hds1))

        # 1. 2nd tokenization using WordPiece
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer

        nlu_tt.append(nlu_tt1)
        l_n.append(len(nlu_tt1))


        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # 2. Generate BERT inputs & indices.
        tokens1, segment_ids1, i_nlu1, i_hds1= generate_inputs(tokenizer, nlu_tt1, hds1,max_seq_length)
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)   ##将token映射为其对应的id（ids是我们训练中真正会用到的数据）

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.   将  token_id mask  segment_ids(0表示question 1表示header) 将这三个数据进行0填充到最大的sequence的长度
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length
        ##批处理的存储数据
        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)


    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    all_encoder_layer, pooled_output = model_bert(all_input_ids, all_segment_ids, all_input_mask)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer, pooled_output, tokens, i_nlu, i_hds, \
           l_n, l_hpu, l_hs, nlu_tt


def get_hql_wemb_n(i_nlu, l_n, hS, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)    ##batch_size
    l_n_max = max(l_n)   ##最大的question的token的长度
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        i_nlu1 = i_nlu[b]
        wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), 0:hS] = all_encoder_layer[b, i_nlu1[0]:i_nlu1[1], :]
    return wemb_n

##i_nlu表示question在token中的位置 l_n表示每个batch里question_token(WTP处理过后)的长度 hS表示隐藏层的状态 num_hiddem_layers表示隐藏层的数量 all_encoder_layers经过表示Bert训练后得到的隐藏层的状态 num_out_layers_n表示输出层有2层，即选择num_hidden_layers最后out_layers的层数
def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)    ##batch_size
    l_n_max = max(l_n)   ##最大的question的token的长度
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        l_n1 = l_n[b]
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), st:ed] = all_encoder_layer[i_layer][b, i_nlu1[0]:i_nlu1[1], :]
    return wemb_n

def generate_xlnet_inputs(tokenizer, nlu1_tok, hds1):
    tokens = []
    segment_ids = []

    i_st_nlu = 0
    for token in nlu1_tok:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)-1

    tokens.append("[SEP]")
    segment_ids.append(0)

    i_hds = []
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    tokens.append("[CLS]")
    segment_ids.append(0)

    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, segment_ids, i_nlu, i_hds

def get_hql_wemb_h(i_hds, l_hpu, l_hs, hS, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), 0:hS] = all_encoder_layer[b, i_hds11[0]:i_hds11[1],:]

    return wemb_h

##i_hds表示每个batch的header在tokens里的位置坐标 l_hpu表示每个header_token的长度 l_hs表示每个batch里header的数量 hS表示hidden的size
def get_wemb_h(i_hds, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, i_hds11[0]:i_hds11[1],:]


    return wemb_h

##
def get_Xlnet_output( nlu_t, hds, max_seq_length,x_model,x_tokenizer):
    """
    Here, input is toknized further by WordPiece (WP) tokenizer and fed into BERT.
    传统词表示方法没有办法很好地处理未知或罕见的词汇。 WordPiece 和 BPE 是两种子词切分算法，两者非常相似。
    INPUT
    :param model_bert:
    :param tokenizer: WordPiece toknizer
    :param nlu: Question
    :param nlu_t: CoreNLP tokenized nlu.
    :param hds: Headers
    :param hs_t: None or 1st-level tokenized headers
    :param max_seq_length: max input token length

    OUTPUT
    tokens: BERT input tokens
    nlu_tt: WP-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.

    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []    ##总Mask

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    nlu_tt = []

    batch_len=0

    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        l_hs.append(len(hds1))

        # 1. 2nd tokenization using WordPiece
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(nlu_t1):
            sub_tokens = x_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        nlu_tt.append(nlu_tt1)
        l_n.append(len(nlu_tt1))

        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP] [CLS]
        # 2. Generate BERT inputs & indices.
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_xlnet_inputs(x_tokenizer, nlu_tt1, hds1)
        input_ids1 = x_tokenizer.convert_tokens_to_ids(tokens1)   ##将token映射为其对应的id（ids是我们训练中真正会用到的数据）

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        ##批处理的存储数据
        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)

        if len(input_ids1)>batch_len:
            batch_len=len(input_ids1)

    ##已目前batch的最大长度作为该batch的长度
    for i,t in enumerate(input_ids):
        while len(t) < batch_len:
            input_ids[i].append(0)
            input_mask[i].append(0)
            segment_ids[i].append(0)
    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    # 4. Generate BERT output.
    # try:
    all_encoder_layer = x_model(all_input_ids, all_input_mask,token_type_ids=all_segment_ids)
    # except RuntimeError:
    #     print(hds)
    #     print(all_input_ids.shape)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer,  tokens, i_nlu, i_hds, l_n, l_hpu, l_hs, nlu_tt


##通过bert做词嵌入  参数描述:bert_config: bert参数  model_bert  tokenizer: bert_vocab  nlut：where_value在question_token的位置 hds:table的表各列的name max_seq_length:最大序列长度 num_out_layers: num_out_layers:最后用于下游任务的bert的层数
def get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=1, num_out_layers_h=1):

    # get contextual output of all tokens from bert
    all_encoder_layer, pooled_output, tokens, i_nlu, i_hds,\
    l_n, l_hpu, l_hs, nlu_tt = get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length)


    ##获得web_hidden用于下游任务
    #wemb_n = get_hql_wemb_n(i_nlu, l_n, bert_config.hidden_size, all_encoder_layer, num_out_layers_n)

    # get the wemb 即在bert_out输出的隐藏层状态里裁剪出num_out_layer的hidden拼接作为每个Question_token(WTP后)的Bert输出
    wemb_n = get_wemb_n(i_nlu, l_n, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_n)
    ##获得header的token的Bert隐藏层状态的拼接作为输出
    #wemb_h = get_hql_wemb_h(x_i_hds,x_l_hpu,x_l_hs,bert_config.hidden_size,xlnet_encoder_layer[0],num_out_layers_n)
    wemb_h = get_wemb_h(i_hds, l_hpu, l_hs, bert_config.hidden_size, bert_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_h)

    return wemb_n, wemb_h, l_n, l_hpu, l_hs,nlu_tt


def gen_pnt_n(g_wvi, mL_w, mL_nt):
    """
    Generate one-hot idx indicating vectors with their lenghts.

    :param g_wvi: e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    where_val idx in nlu_t. 0 = <BEG>, -1 = <END>.
    :param mL_w: 4
    :param mL_nt: 200
    :return:
    """
    bS = len(g_wvi)
    for g_wvi1 in g_wvi:
        for g_wvi11 in g_wvi1:
            l11 = len(g_wvi11)

    mL_g_wvi = max([max([0] + [len(tok) for tok in gwsi]) for gwsi in g_wvi]) - 1
    # zero because of '' case.
    # -1 because we already have <BEG>
    if mL_g_wvi < 1:
        mL_g_wvi = 1
    # NLq_token_pos = torch.zeros(bS, 5 - 1, mL_g_wvi, self.max_NLq_token_num)

    # l_g_wvi = torch.zeros(bS, 5 - 1)
    pnt_n = torch.zeros(bS, mL_w, mL_g_wvi, mL_nt).to(device) # one hot
    l_g_wvi = torch.zeros(bS, mL_w).to(device)

    for b, g_wvi1 in enumerate(g_wvi):
        i_wn = 0  # To prevent error from zero number of condition.
        for i_wn, g_wvi11 in enumerate(g_wvi1):
            # g_wvi11: [0, where_conds pos in NLq, end]
            g_wvi11_n1 = g_wvi11[:-1]  # doesn't count <END> idx.
            l_g_wvi[b, i_wn] = len(g_wvi11_n1)
            for t, idx in enumerate(g_wvi11_n1):
                pnt_n[b, i_wn, t, idx] = 1

            # Pad
        if i_wn < (mL_w - 1):  # maximum number of conidtions is 4
            pnt_n[b, i_wn + 1:, 0, 1] = 1  # # cannot understand... [<BEG>, <END>]??
            l_g_wvi[b, i_wn + 1:] = 1  # it means there is only <BEG>.


    return pnt_n, l_g_wvi


def pred_sc(s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc = []
    for s_sc1 in s_sc:
        pr_sc.append(s_sc1.argmax().item())

    return pr_sc

def pred_sc_beam(s_sc, beam_size):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc_beam = []


    for s_sc1 in s_sc:
        val, idxes = s_sc1.topk(k=beam_size)
        pr_sc_beam.append(idxes.tolist())

    return pr_sc_beam

def pred_sa(s_sa):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sa = []
    for s_sa1 in s_sa:
        pr_sa.append(s_sa1.argmax().item())

    return pr_sa


def pred_wn(s_wn):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wn = []
    for s_wn1 in s_wn:
        pr_wn.append(s_wn1.argmax().item())
        # print(pr_wn, s_wn1)
        # if s_wn1.argmax().item() == 3:
        #     input('')

    return pr_wn

def pred_wc_old(sql_i, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wc = []
    for b, sql_i1 in enumerate(sql_i):
        wn = len(sql_i1['conds'])
        s_wc1 = s_wc[b]

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn]
        pr_wc1.sort()

        pr_wc.append(list(pr_wc1))
    return pr_wc

def pred_wc(wn, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    pr_wc = []
    for b, wn1 in enumerate(wn):
        s_wc1 = s_wc[b]

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn1]
        pr_wc1.sort()

        pr_wc.append(list(pr_wc1))
    return pr_wc

def pred_wc_sorted_by_prob(s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted by prob.
    All colume-indexes are returned here.
    """
    # get g_num
    bS = len(s_wc)
    pr_wc = []

    for b in range(bS):
        s_wc1 = s_wc[b]
        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())
        pr_wc.append(list(pr_wc1))
    return pr_wc


def pred_wo(wn, s_wo):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # s_wo = [B, 4, n_op]
    pr_wo_a = s_wo.argmax(dim=2)  # [B, 4]
    # get g_num
    pr_wo = []
    for b, pr_wo_a1 in enumerate(pr_wo_a):
        wn1 = wn[b]
        pr_wo.append(list(pr_wo_a1.data.cpu().numpy()[:wn1]))

    return pr_wo


def pred_wvi_se(wn, s_wv):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx
    """

    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)  # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]

    s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    pr_wvi_st_idx = s_wv_st.argmax(dim=2) # [B, 4, mL] -> [B, 4, 1]
    pr_wvi_ed_idx = s_wv_ed.argmax(dim=2)

    pr_wvi = []
    for b, wn1 in enumerate(wn):
        pr_wvi1 = []
        for i_wn in range(wn1):
            pr_wvi_st_idx11 = pr_wvi_st_idx[b][i_wn]
            pr_wvi_ed_idx11 = pr_wvi_ed_idx[b][i_wn]
            pr_wvi1.append([pr_wvi_st_idx11.item(), pr_wvi_ed_idx11.item()])
        pr_wvi.append(pr_wvi1)

    return pr_wvi

def pred_wvi_se_beam(max_wn, s_wv, beam_size):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx


    output:
    pr_wvi_beam = [B, max_wn, n_pairs, 2]. 2 means [st, ed].
    prob_wvi_beam = [B, max_wn, n_pairs]
    """
    bS = s_wv.shape[0]

    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)  # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]

    s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    prob_wv_st = F.softmax(s_wv_st, dim=-1).detach().to('cpu').numpy()
    prob_wv_ed = F.softmax(s_wv_ed, dim=-1).detach().to('cpu').numpy()

    k_logit = int(ceil(sqrt(beam_size)))
    n_pairs = k_logit**2
    assert n_pairs >= beam_size
    values_st, idxs_st = s_wv_st.topk(k_logit) # [B, 4, mL] -> [B, 4, k_logit]
    values_ed, idxs_ed = s_wv_ed.topk(k_logit) # [B, 4, mL] -> [B, 4, k_logit]

    # idxs = [B, k_logit, 2]
    # Generate all possible combination of st, ed indices & prob
    pr_wvi_beam = [] # [B, max_wn, k_logit**2 [st, ed] paris]
    prob_wvi_beam = zeros([bS, max_wn, n_pairs])
    for b in range(bS):
        pr_wvi_beam1 = []

        idxs_st1 = idxs_st[b]
        idxs_ed1 = idxs_ed[b]
        for i_wn in range(max_wn):
            idxs_st11 = idxs_st1[i_wn]
            idxs_ed11 = idxs_ed1[i_wn]

            pr_wvi_beam11 = []
            pair_idx = -1
            for i_k in range(k_logit):
                for j_k in range(k_logit):
                    pair_idx += 1
                    st = idxs_st11[i_k].item()
                    ed = idxs_ed11[j_k].item()
                    pr_wvi_beam11.append([st, ed])

                    p1 = prob_wv_st[b, i_wn, st]
                    p2 = prob_wv_ed[b, i_wn, ed]
                    prob_wvi_beam[b, i_wn, pair_idx] = p1*p2
            pr_wvi_beam1.append(pr_wvi_beam11)
        pr_wvi_beam.append(pr_wvi_beam1)


    # prob

    return pr_wvi_beam, prob_wvi_beam

def is_whitespace_g_wvi(c):
    # if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    if c == " ":
        return True
    return False

def convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_wp_t, wp_to_wh_index, nlu):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str_wp = [] # word-piece version
    pr_wv_str = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        pr_wv_str_wp1 = []
        pr_wv_str1 = []
        wp_to_wh_index1 = wp_to_wh_index[b]
        nlu_wp_t1 = nlu_wp_t[b]
        nlu_t1 = nlu_t[b]

        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st_idx, ed_idx = pr_wvi11

            # Ad-hoc modification of ed_idx to deal with wp-tokenization effect.
            # e.g.) to convert "butler cc (" ->"butler cc (ks)" (dev set 1st question).
            pr_wv_str_wp11 = nlu_wp_t1[st_idx:ed_idx+1]
            pr_wv_str_wp1.append(pr_wv_str_wp11)

            st_wh_idx = wp_to_wh_index1[st_idx]
            ed_wh_idx = wp_to_wh_index1[ed_idx]
            pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx+1]

            pr_wv_str1.append(pr_wv_str11)

        pr_wv_str_wp.append(pr_wv_str_wp1)
        pr_wv_str.append(pr_wv_str1)

    return pr_wv_str, pr_wv_str_wp




def pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo):
    pr_sc = pred_sc(s_sc)
    pr_sa = pred_sa(s_sa)
    pr_wn = pred_wn(s_wn)
    pr_wc = pred_wc(pr_wn, s_wc)
    pr_wo = pred_wo(pr_wn, s_wo)

    return pr_sc, pr_sa, pr_wn, pr_wc, pr_wo





def merge_wv_t1_eng(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    nlq = NLq.lower()
    where_str_tokens = [tok.lower() for tok in where_str_tokens]
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$'
    special = {'-LRB-': '(',
               '-RRB-': ')',
               '-LSB-': '[',
               '-RSB-': ']',
               '``': '"',
               '\'\'': '"',
               }
               # '--': '\u2013'} # this generate error for test 5661 case.
    ret = ''
    double_quote_appear = 0
    for raw_w_token in where_str_tokens:
        # if '' (empty string) of None, continue
        if not raw_w_token:
            continue

        # Change the special characters
        w_token = special.get(raw_w_token, raw_w_token)  # maybe necessary for some case?

        # check the double quote
        if w_token == '"':
            double_quote_appear = 1 - double_quote_appear

        # Check whether ret is empty. ret is selected where condition.
        if len(ret) == 0:
            pass
        # Check blank character.
        elif len(ret) > 0 and ret + ' ' + w_token in nlq:
            # Pad ' ' if ret + ' ' is part of nlq.
            ret = ret + ' '

        elif len(ret) > 0 and ret + w_token in nlq:
            pass  # already in good form. Later, ret + w_token will performed.

        # Below for unnatural question I guess. Is it likely to appear?
        elif w_token == '"':
            if double_quote_appear:
                ret = ret + ' '  # pad blank line between next token when " because in this case, it is of closing apperas
                # for the case of opening, no blank line.

        elif w_token[0] not in alphabet:
            pass  # non alphabet one does not pad blank line.

        # when previous character is the special case.
        elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) and (ret[-1] != '"' or not double_quote_appear):
            ret = ret + ' '
        ret = ret + w_token

    return ret.strip()



def find_sql_where_op(gt_sql_tokens_part):
    """
    gt_sql_tokens_part: Between 'WHERE' and 'AND'(if exists).
    """
    # sql_where_op = ['=', 'EQL', '<', 'LT', '>', 'GT']
    sql_where_op = ['EQL','LT','GT'] # wv sometimes contains =, < or >.


    for sql_where_op in sql_where_op:
        if sql_where_op in gt_sql_tokens_part:
            found_sql_where_op = sql_where_op
            break

    return found_sql_where_op


def find_sub_list(sl, l):
    # from stack overflow.
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results

def get_g_wvi_bert(nlu, nlu_t, wh_to_wp_index, sql_i, sql_t, tokenizer, nlu_wp_t):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, sql_i1 in enumerate(sql_i):
        nlu1 = nlu[b]
        nlu_t1 = nlu_t[b]
        nlu_wp_t1 = nlu_wp_t[b]
        sql_t1 = sql_t[b]
        wh_to_wp_index1 = wh_to_wp_index[b]

        st = sql_t1.index('WHERE') + 1 if 'WHERE' in sql_t1 else len(sql_t1)
        g_wvi1 = []
        while st < len(sql_t1):
            if 'AND' not in sql_t1[st:]:
                ed = len(sql_t1)
            else:
                ed = sql_t1[st:].index('AND') + st
            sql_wop = find_sql_where_op(sql_t1[st:ed])  # sql where operator
            st_wop = st + sql_t1[st:ed].index(sql_wop)

            wv_str11_t = sql_t1[st_wop + 1:ed]
            results = find_sub_list(wv_str11_t, nlu_t1)
            st_idx, ed_idx = results[0]

            st_wp_idx = wh_to_wp_index1[st_idx]
            ed_wp_idx = wh_to_wp_index1[ed_idx]


            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)
            st = ed + 1
        g_wvi.append(g_wvi1)

    return g_wvi

##获得value_token在token中的起始到终止的位置
def get_g_wvi_bert_from_g_wvi_corenlp(wh_to_wp_index, g_wvi_corenlp):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, g_wvi_corenlp1 in enumerate(g_wvi_corenlp):
        wh_to_wp_index1 = wh_to_wp_index[b]
        g_wvi1 = []
        for i_wn, g_wvi_corenlp11 in enumerate(g_wvi_corenlp1):

            st_idx, ed_idx = g_wvi_corenlp11

            st_wp_idx = wh_to_wp_index1[st_idx]
            ed_wp_idx = wh_to_wp_index1[ed_idx]

            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)

        g_wvi.append(g_wvi1)

    return g_wvi


def get_g_wvi_bert_from_sql_i(nlu, nlu_t, wh_to_wp_index, sql_i, sql_t, tokenizer, nlu_wp_t):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, sql_i1 in enumerate(sql_i):
        nlu1 = nlu[b]
        nlu_t1 = nlu_t[b]
        nlu_wp_t1 = nlu_wp_t[b]
        sql_t1 = sql_t[b]
        wh_to_wp_index1 = wh_to_wp_index[b]

        st = sql_t1.index('WHERE') + 1 if 'WHERE' in sql_t1 else len(sql_t1)
        g_wvi1 = []
        while st < len(sql_t1):
            if 'AND' not in sql_t1[st:]:
                ed = len(sql_t1)
            else:
                ed = sql_t1[st:].index('AND') + st
            sql_wop = find_sql_where_op(sql_t1[st:ed])  # sql where operator
            st_wop = st + sql_t1[st:ed].index(sql_wop)

            wv_str11_t = sql_t1[st_wop + 1:ed]
            results = find_sub_list(wv_str11_t, nlu_t1)
            st_idx, ed_idx = results[0]

            st_wp_idx = wh_to_wp_index1[st_idx]
            ed_wp_idx = wh_to_wp_index1[ed_idx]


            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)
            st = ed + 1
        g_wvi.append(g_wvi1)

    return g_wvi

def get_cnt_sc(g_sc, pr_sc):
    cnt = 0
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt += 1

    return cnt

def get_cnt_sc_list(g_sc, pr_sc):
    cnt_list = []
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list

def get_cnt_sa(g_sa, pr_sa):
    cnt = 0
    for b, g_sa1 in enumerate(g_sa):
        pr_sa1 = pr_sa[b]
        if pr_sa1 == g_sa1:
            cnt += 1

    return cnt


def get_cnt_wn(g_wn, pr_wn):
    cnt = 0
    for b, g_wn1 in enumerate(g_wn):
        pr_wn1 = pr_wn[b]
        if pr_wn1 == g_wn1:
            cnt += 1

    return cnt

def get_cnt_wc(g_wc, pr_wc):
    cnt = 0
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            continue
        else:
            wc1 = array(g_wc1)
            wc1.sort()

            if array_equal(pr_wc1, wc1):
                cnt += 1

    return cnt

def get_cnt_wc_list(g_wc, pr_wc):
    cnt_list= []
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            cnt_list.append(0)
            continue
        else:
            wc1 = array(g_wc1)
            wc1.sort()

            if array_equal(pr_wc1, wc1):
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list


def get_cnt_wo(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt = 0
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            continue
        else:
            # Sort based on wc sequence.
            if mode == 'test':
                idx = argsort(array(g_wc1))

                g_wo1_s = array(g_wo1)[idx]
                g_wo1_s = list(g_wo1_s)
            elif mode == 'train':
                # due to teacher forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt += 1
    return cnt

def get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt_list=[]
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            # Sort based wc sequence.
            if mode == 'test':
                idx = argsort(array(g_wc1))

                g_wo1_s = array(g_wo1)[idx]
                g_wo1_s = list(g_wo1_s)
            elif mode == 'train':
                # due to tearch forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list


def get_cnt_wv(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv

    g_wvi
    """
    cnt = 0
    for b, g_wvi1 in enumerate(g_wvi):
        pr_wvi1 = pr_wvi[b]
        g_wc1 = g_wc[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt += 1

    return cnt


def get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wvi1 in enumerate(g_wvi):
        g_wc1 = g_wc[b]
        pr_wvi1 = pr_wvi[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list


def get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wc1 in enumerate(g_wc):
        pr_wn1 = len(pr_sql_i[b]["conds"])
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = argsort(array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi_str11 = str(g_sql_i[b]["where"][idx11][2]).lower()
                pr_wvi_str11 = str(pr_sql_i[b]["where"][i_wn][2]).lower()
                # print(g_wvi_str11)
                # print(pr_wvi_str11)
                # print(g_wvi_str11==pr_wvi_str11)
                if g_wvi_str11 != pr_wvi_str11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_sw(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sc = get_cnt_sc(g_sc, pr_sc)
    cnt_sa = get_cnt_sa(g_sa, pr_sa)
    cnt_wn = get_cnt_wn(g_wn, pr_wn)
    cnt_wc = get_cnt_wc(g_wc, pr_wc)
    cnt_wo = get_cnt_wo(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)
    cnt_wv = get_cnt_wv(g_wn, g_wc, g_wvi, pr_wvi, mode)

    return cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv

def get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, g_sql_i, pr_sql_i,mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sc = get_cnt_sc_list(g_sc, pr_sc)
    cnt_sa = get_cnt_sc_list(g_sa, pr_sa)
    cnt_wn = get_cnt_sc_list(g_wn, pr_wn)
    cnt_wc = get_cnt_wc_list(g_wc, pr_wc)
    cnt_wo = get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)

    #cnt_wv = get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode) # compare using wv-str which presented in original data.


    return cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo


def get_cnt_lx_list(cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1):
    # all cnt are list here.
    cnt_list = []
    for csc, csa, cwn, cwc, cwo in zip(cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1):
        if csc and csa and cwn and cwc and cwo:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list


def get_cnt_x_list(engine, tb, g_sc, g_sa, g_sql_i, pr_sc, pr_sa, pr_sql_i):
    cnt_x1_list = []
    g_ans = []
    pr_ans = []
    for b in range(len(g_sc)):
        g_ans1 = engine.execute(tb[b]['id'], g_sc[b], g_sa[b], g_sql_i[b]['conds'])
        # print(f'cnt: {cnt}')
        # print(f"pr_sql_i: {pr_sql_i[b]['conds']}")
        try:
            pr_ans1 = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], pr_sql_i[b]['conds'])

            if bool(pr_ans1):  # not empty due to lack of the data from incorretly generated sql
                if g_ans1 == pr_ans1:
                    cnt_x1 = 1
                else:
                    cnt_x1 = 0
            else:
                cnt_x1 = 0
        except:
            # type error etc... Execution-guided decoding may be used here.
            pr_ans1 = None
            cnt_x1 = 0
        cnt_x1_list.append(cnt_x1)
        g_ans.append(g_ans1)
        pr_ans.append(pr_ans1)

    return cnt_x1_list, g_ans, pr_ans

def get_mean_grad(named_parameters):
    """
    Get list of mean, std of grad of each parameters
    Code based on web searched result..
    """
    mu_list = []
    sig_list = []
    for name, param in named_parameters:
        if param.requires_grad: # and ("bias" not in name) :
            # bias makes std = nan as it is of single parameters
            magnitude = param.grad.abs()
            mu_list.append(magnitude.mean())
            if len(magnitude) == 1:
                # why nan for single param? Anyway to avoid that..
                sig_list.append(torch.tensor(0))
            else:
                sig_list.append(magnitude.std())

            # if "svp_se"

    return mu_list, sig_list


def generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, nlu):
    pr_sql_i = []
    for b, nlu1 in enumerate(nlu):
        conds = []
        for i_wn in range(pr_wn[b]):
            conds1 = []
            conds1.append(pr_wc[b][i_wn])
            conds1.append(pr_wo[b][i_wn])
            conds.append(conds1)

        pr_sql_i1 = {'agg': pr_sa[b], 'sel': pr_sc[b], 'conds': conds}
        pr_sql_i.append(pr_sql_i1)
    return pr_sql_i


def save_for_evaluation(path_save, results, dset_name, ):
    path_save_file = os.path.join(path_save, f'results_{dset_name}.jsonl')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)

def save_for_evaluation_aux(path_save, results, dset_name, ):
    path_save_file = os.path.join(path_save, f'results_aux_{dset_name}.jsonl')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'

            f.writelines(json_str)


def check_sc_sa_pairs(tb, pr_sc, pr_sa, ):
    """
    Check whether pr_sc, pr_sa are allowed pairs or not.
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

    """
    bS = len(pr_sc)
    check = [False] * bS
    for b, pr_sc1 in enumerate(pr_sc):
        pr_sa1 = pr_sa[b]
        hd_types1 = tb[b]['types']
        hd_types11 = hd_types1[pr_sc1]
        if hd_types11 == 'text':
            if pr_sa1 == 0 or pr_sa1 == 3: # ''
                check[b] = True
            else:
                check[b] = False

        elif hd_types11 == 'real':
            check[b] = True
        else:
            raise Exception("New TYPE!!")

    return check


def remap_sc_idx(idxs, pr_sc_beam):
    for b, idxs1 in enumerate(idxs):
        for i_beam, idxs11 in enumerate(idxs1):
            sc_beam_idx = idxs[b][i_beam][0]
            sc_idx = pr_sc_beam[b][sc_beam_idx]
            idxs[b][i_beam][0] = sc_idx

    return idxs


def sort_and_generate_pr_w(pr_sql_i):
    pr_wc = []
    pr_wo = []
    pr_wv = []
    for b, pr_sql_i1 in enumerate(pr_sql_i):
        conds1 = pr_sql_i1["conds"]
        pr_wc1 = []
        pr_wo1 = []
        pr_wv1 = []

        # Generate
        for i_wn, conds11 in enumerate(conds1):
            pr_wc1.append( conds11[0])
            pr_wo1.append( conds11[1])
            pr_wv1.append( conds11[2])

        # sort based on pr_wc1
        idx = argsort(pr_wc1)
        pr_wc1 = array(pr_wc1)[idx].tolist()
        pr_wo1 = array(pr_wo1)[idx].tolist()
        pr_wv1 = array(pr_wv1)[idx].tolist()

        conds1_sorted = []
        for i, idx1 in enumerate(idx):
            conds1_sorted.append( conds1[idx1] )


        pr_wc.append(pr_wc1)
        pr_wo.append(pr_wo1)
        pr_wv.append(pr_wv1)

        pr_sql_i1['conds'] = conds1_sorted

    return pr_wc, pr_wo, pr_wv, pr_sql_i

def generate_sql_q(sql_i, tb):
    sql_q = []
    for b, sql_i1 in enumerate(sql_i):
        tb1 = tb[b]
        sql_q1 = generate_sql_q1(sql_i1, tb1)
        sql_q.append(sql_q1)

    return sql_q

def generate_sql_q1(sql_i1, tb1):
    """
        sql = {'sel': 5, 'agg': 4, 'conds': [[3, 0, '59']]}
        agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
        cond_ops = ['=', '>', '<', 'OP']

        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query

        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
    cond_ops = ['=', '>', '<', 'OP']

    headers = tb1["header"]
    # select_header = headers[sql['sel']].lower()
    # try:
    #     select_table = tb1["name"]
    # except:
    #     print(f"No table name while headers are {headers}")
    select_table = tb1["id"]

    select_agg = agg_ops[sql_i1['agg']]
    select_header = headers[sql_i1['sel']]
    sql_query_part1 = f'SELECT {select_agg}({select_header}) '


    where_num = len(sql_i1['conds'])
    if where_num == 0:
        sql_query_part2 = f'FROM {select_table}'
        # sql_plus_query_part2 = f'FROM {select_table}'

    else:
        sql_query_part2 = f'FROM {select_table} WHERE'
        # sql_plus_query_part2 = f'FROM {select_table_refined} WHERE'
        # ----------------------------------------------------------------------------------------------------------
        for i in range(where_num):
            # check 'OR'
            # number_of_sub_conds = len(sql['conds'][i])
            where_header_idx, where_op_idx, where_str = sql_i1['conds'][i]
            where_header = headers[where_header_idx]
            where_op = cond_ops[where_op_idx]
            if i > 0:
                sql_query_part2 += ' AND'
                # sql_plus_query_part2 += ' AND'

            sql_query_part2 += f" {where_header} {where_op} {where_str}"

    sql_query = sql_query_part1 + sql_query_part2
    # sql_plus_query = sql_plus_query_part1 + sql_plus_query_part2

    return sql_query


def get_pnt_idx1(col_pool_type, st_ed):
    st, ed = st_ed
    if col_pool_type == 'start_tok':
        pnt_idx1 = st
    elif col_pool_type == 'end_tok':
        pnt_idx1 = ed
    elif col_pool_type == 'avg':
        pnt_idx1 = arange(st, ed, 1)
    return pnt_idx1


def gen_g_pnt_idx(g_wvi, sql_i, i_hds, i_sql_vocab, col_pool_type):
    """
    sql_vocab = (
        0.. "sql none", "sql max", "sql min", "sql count", "sql sum", "sql average", ..5
        6.. "sql select", "sql where", "sql and", .. 8
        9.. "sql equal", "sql greater than", "sql less than", .. 11
        12.. "sql start", "sql end" .. 13
    )
    """
    g_pnt_idxs = []



    for b, sql_i1 in enumerate(sql_i):
        i_sql_vocab1 = i_sql_vocab[b]
        i_hds1 = i_hds[b]
        g_pnt_idxs1 = []

        # start token
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[-2])
        g_pnt_idxs1.append(pnt_idx1)

        # select token
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[6])
        g_pnt_idxs1.append(pnt_idx1)

        # select agg
        idx_agg = sql_i1["agg"]
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[idx_agg])
        g_pnt_idxs1.append(pnt_idx1)

        # select column
        idx_sc = sql_i1["sel"]
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_hds1[idx_sc])
        g_pnt_idxs1.append(pnt_idx1)

        conds = sql_i1["conds"]
        wn = len(conds)
        if wn <= 0:
            pass
        else:
            # select where
            pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[7])
            g_pnt_idxs1.append(pnt_idx1)

            for i_wn, conds1 in enumerate(conds):
                # where column
                idx_wc = conds1[0]
                pnt_idx1 = get_pnt_idx1(col_pool_type, i_hds1[idx_wc])
                g_pnt_idxs1.append(pnt_idx1)

                # where op
                idx_wo = conds1[1]
                pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[idx_wo + 9])
                g_pnt_idxs1.append(pnt_idx1)

                # where val
                st, ed = g_wvi[b][i_wn]
                end_pos_of_sql_vocab = i_sql_vocab1[-1][-1]
                g_pnt_idxs1.append(st + 1 + end_pos_of_sql_vocab)  # due to inital [CLS] token in BERT-input vector
                g_pnt_idxs1.append(ed + 1 + end_pos_of_sql_vocab)  # due to inital [CLS] token in BERT-input vector

                # and token
                if i_wn < wn - 1:
                    pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[8])
                    g_pnt_idxs1.append(pnt_idx1)

        # end token
        pnt_idx1 = get_pnt_idx1(col_pool_type, i_sql_vocab1[-1])
        g_pnt_idxs1.append(pnt_idx1)

        g_pnt_idxs.append(g_pnt_idxs1)

    return g_pnt_idxs


def pred_pnt_idxs(score, pnt_start_tok, pnt_end_tok):
    pr_pnt_idxs = []
    for b, score1 in enumerate(score):
        # score1 = [T, max_seq_length]
        pr_pnt_idxs1 = [pnt_start_tok]
        for t, score11 in enumerate(score1):
            pnt = score11.argmax().item()
            pr_pnt_idxs1.append(pnt)

            if pnt == pnt_end_tok:
                break
        pr_pnt_idxs.append(pr_pnt_idxs1)

    return pr_pnt_idxs


def generate_sql_q_s2s(pnt_idxs, tokens, tb):
    sql_q = []
    for b, pnt_idxs1 in enumerate(pnt_idxs):
        tb1 = tb[b]
        sql_q1 = generate_sql_q1_s2s(pnt_idxs1, tokens[b], tb1)
        sql_q.append(sql_q1)

    return sql_q


def generate_sql_q1_s2s(pnt_idxs1, tokens1, tb1):
    """
        agg_ops = ['', 'max', 'min', 'count', 'sum', 'avg']
        cond_ops = ['=', '>', '<', 'OP']

        Temporal as it can show only one-time conditioned case.
        sql_query: real sql_query
        sql_plus_query: More redable sql_query

        "PLUS" indicates, it deals with the some of db specific facts like PCODE <-> NAME
    """
    sql_query = ""
    for t, pnt_idxs11 in enumerate(pnt_idxs1):
        tok = tokens1[pnt_idxs11]
        sql_query += tok
        if t < len(pnt_idxs1)-1:
            sql_query += " "


    return sql_query


# Generate sql_i from pnt_idxs
def find_where_pnt_belong(pnt, vg):
    idx_sub = -1
    for i, st_ed in enumerate(vg):
        st, ed = st_ed
        if pnt < ed and pnt >= st:
            idx_sub = i

    return idx_sub


def gen_pnt_i_from_pnt(pnt, i_sql_vocab1, i_nlu1, i_hds1):
    # Find where it belong
    vg_list = [i_sql_vocab1, [i_nlu1], i_hds1] # as i_nlu has only single st and ed
    i_vg = -1
    i_vg_sub = -1
    for i, vg in enumerate(vg_list):
        idx_sub = find_where_pnt_belong(pnt, vg)
        if idx_sub > -1:
            i_vg = i
            i_vg_sub = idx_sub
            break
    return i_vg, i_vg_sub


def gen_i_vg_from_pnt_idxs(pnt_idxs, i_sql_vocab, i_nlu, i_hds):
    i_vg_list = []
    i_vg_sub_list = []
    for b, pnt_idxs1 in enumerate(pnt_idxs):
        # if properly generated,
        sql_q1_list = []
        i_vg_list1 = [] # index of (sql_vocab, nlu, hds)
        i_vg_sub_list1 = [] # index inside of each vocab group

        for t, pnt in enumerate(pnt_idxs1):
            i_vg, i_vg_sub = gen_pnt_i_from_pnt(pnt, i_sql_vocab[b], i_nlu[b], i_hds[b])
            i_vg_list1.append(i_vg)
            i_vg_sub_list1.append(i_vg_sub)

        # sql_q1 = sql_q1.join(' ')
        # sql_q.append(sql_q1)
        i_vg_list.append(i_vg_list1)
        i_vg_sub_list.append(i_vg_sub_list1)
    return i_vg_list, i_vg_sub_list


def gen_sql_q_from_i_vg(tokens, nlu, nlu_t, hds, tt_to_t_idx, pnt_start_tok, pnt_end_tok, pnt_idxs, i_vg_list, i_vg_sub_list):
    """
    (
        "none", "max", "min", "count", "sum", "average",
        "select", "where", "and",
        "equal", "greater than", "less than",
        "start", "end"
    ),
    """
    sql_q = []
    sql_i = []
    for b, nlu_t1 in enumerate(nlu_t):
        sql_q1_list = []
        sql_i1 = {}
        tt_to_t_idx1 = tt_to_t_idx[b]
        nlu_st_observed = False
        agg_observed = False
        wc_obs = False
        wo_obs = False
        conds = []

        for t, i_vg in enumerate(i_vg_list[b]):
            i_vg_sub = i_vg_sub_list[b][t]
            pnt = pnt_idxs[b][t]
            if i_vg == 0:
                # sql_vocab
                if pnt == pnt_start_tok or pnt == pnt_end_tok:
                    pass
                else:
                    tok = tokens[b][pnt]
                    if tok in ["none", "max", "min", "count", "sum", "average"]:
                        agg_observed = True
                        if tok == "none":
                            pass
                        sql_i1["agg"] = ["none", "max", "min", "count", "sum", "average"].index(tok)
                    else:
                        if tok in ["greater", "less", "equal"]:
                            if tok == 'greater':
                                tok = '>'
                            elif tok == 'less':
                                tok = '<'
                            elif tok == 'equal':
                                tok = '='

                            # gen conds1
                            if wc_obs:
                                conds1.append( ['=','>','<'].index(tok) )
                                wo_obs = True

                        sql_q1_list.append(tok)

            elif i_vg == 1:
                # nlu case
                if not nlu_st_observed:
                    idx_nlu_st = pnt
                    nlu_st_observed = True
                else:
                    # now to wrap up
                    idx_nlu_ed = pnt
                    st_wh_idx = tt_to_t_idx1[idx_nlu_st - pnt_end_tok - 2]
                    ed_wh_idx = tt_to_t_idx1[idx_nlu_ed - pnt_end_tok - 2]
                    pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx + 1]
                    merged_wv11 = merge_wv_t1_eng(pr_wv_str11, nlu[b])
                    sql_q1_list.append(merged_wv11)
                    nlu_st_observed = False

                    if wc_obs and wo_obs:
                        conds1.append(merged_wv11)
                        conds.append(conds1)

                        wc_obs = False
                        wo_obs = False


            elif i_vg == 2:
                # headers
                tok = hds[b][i_vg_sub]
                if agg_observed:
                    sql_q1_list.append(f"({tok})")
                    sql_i1["sel"] = i_vg_sub
                    agg_observed = False
                else:
                    wc_obs = True
                    conds1 = [i_vg_sub]

                    sql_q1_list.append(tok)

        # insert table name between.
        sql_i1["conds"] = conds
        sql_i.append(sql_i1)
        sql_q1 = ' '.join(sql_q1_list)
        sql_q.append(sql_q1)

    return sql_q, sql_i


def get_cnt_lx_list_s2s(g_pnt_idxs, pr_pnt_idxs):
    # all cnt are list here.
    cnt_list = []
    for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
        pr_pnt_idxs1 = pr_pnt_idxs[b]

        if g_pnt_idxs1 == pr_pnt_idxs1:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list


def get_wemb_h_FT_Scalar_1(i_hds, l_hs, hS, all_encoder_layer, col_pool_type='start_tok'):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]

    # i_hds = [ [  Batch 1 ] [  Batch 2  ] ]
    #  [Batch 1] = [ (col1_st_idx, col1_ed_idx), (col2_st_idx, col2_ed_idx), ...]
    # i_hds = [[(11, 14), (15, 19), (20, 21), (22, 24), (25, 27), (28, 29)],
            #  [(16, 19), (20, 24), (25, 26), (27, 29), (30, 32), (33, 34)]]

    pool_type = 'start_tok', 'end_tok', 'avg'

    """
    bS = len(l_hs)
    l_hs_max = max(l_hs)
    wemb_h = torch.zeros([bS, l_hs_max, hS]).to(device)
    for b, i_hds1 in enumerate(i_hds):
        for i_hd, st_ed_pair in enumerate(i_hds1):
            st, ed = st_ed_pair
            if col_pool_type == 'start_tok':
                vec = all_encoder_layer[-1][b, st,:]
            elif col_pool_type == 'end_tok':
                vec = all_encoder_layer[-1][b, ed, :]
            elif col_pool_type == 'avg':
                vecs = all_encoder_layer[-1][b, st:ed,:]
                vec = vecs.mean(dim=1, keepdim=True)
            else:
                raise ValueError
            wemb_h[b, i_hd, :] = vec

    return wemb_h


def cal_prob(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi):
    """

    :param s_sc: [B, l_h]
    :param s_sa: [B, l_a] # 16
    :param s_wn: [B, 5]
    :param s_wc: [B, l_h]
    :param s_wo: [B, 4, l_o] #
    :param s_wv: [B, 4, 22]
    :return:
    """
    # First get selected index

    #

    # Predict prob
    p_sc = cal_prob_sc(s_sc, pr_sc)
    p_sa = cal_prob_sa(s_sa, pr_sa)
    p_wn = cal_prob_wn(s_wn, pr_wn)
    p_wc = cal_prob_wc(s_wc, pr_wc)
    p_wo = cal_prob_wo(s_wo, pr_wo)
    p_wvi = cal_prob_wvi_se(s_wv, pr_wvi)

    # calculate select-clause probability
    p_select = cal_prob_select(p_sc, p_sa)

    # calculate where-clause probability
    p_where  = cal_prob_where(p_wn, p_wc, p_wo, p_wvi)

    # calculate total probability
    p_tot = cal_prob_tot(p_select, p_where)

    return p_tot, p_select, p_where, p_sc, p_sa, p_wn, p_wc, p_wo, p_wvi

def cal_prob_tot(p_select, p_where):
    p_tot = []
    for b, p_select1 in enumerate(p_select):
        p_where1 = p_where[b]
        p_tot.append( p_select1 * p_where1 )

    return p_tot

def cal_prob_select(p_sc, p_sa):
    p_select = []
    for b, p_sc1 in enumerate(p_sc):
        p1 = 1.0
        p1 *= p_sc1
        p1 *= p_sa[b]

        p_select.append(p1)
    return p_select

def cal_prob_where(p_wn, p_wc, p_wo, p_wvi):
    p_where = []
    for b, p_wn1 in enumerate(p_wn):
        p1 = 1.0
        p1 *= p_wn1
        p_wc1 = p_wc[b]

        for i_wn, p_wc11 in enumerate(p_wc1):
            p_wo11 = p_wo[b][i_wn]
            p_wv11_st, p_wv11_ed = p_wvi[b][i_wn]

            p1 *= p_wc11
            p1 *= p_wo11
            p1 *= p_wv11_st
            p1 *= p_wv11_ed

        p_where.append(p1)

    return p_where


def cal_prob_sc(s_sc, pr_sc):
    ps = F.softmax(s_sc, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_sc1 = pr_sc[b]
        p1 = ps1[pr_sc1]
        p.append(p1.item())

    return p

def cal_prob_sa(s_sa, pr_sa):
    ps = F.softmax(s_sa, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_sa1 = pr_sa[b]
        p1 = ps1[pr_sa1]
        p.append(p1.item())

    return p

def cal_prob_wn(s_wn, pr_wn):
    ps = F.softmax(s_wn, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_wn1 = pr_wn[b]
        p1 = ps1[pr_wn1]
        p.append(p1.item())

    return p

def cal_prob_wc(s_wc, pr_wc):
    ps = torch.sigmoid(s_wc)
    ps_out = []
    for b, pr_wc1 in enumerate(pr_wc):
        ps1 = array(ps[b].cpu())
        ps_out1 = ps1[pr_wc1]
        ps_out.append(list(ps_out1))

    return ps_out

def cal_prob_wo(s_wo, pr_wo):
    # assume there is always at least single condition.
    ps = F.softmax(s_wo, dim=2)
    ps_out = []


    for b, pr_wo1 in enumerate(pr_wo):
        ps_out1 = []
        for n, pr_wo11 in enumerate(pr_wo1):
            ps11 = ps[b][n]
            ps_out1.append( ps11[pr_wo11].item() )


        ps_out.append(ps_out1)

    return ps_out


def cal_prob_wvi_se(s_wv, pr_wvi):
    prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()
    p_wv = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        p_wv1 = []
        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st, ed = pr_wvi11
            p_st = prob_wv[b, i_wn, st, 0]
            p_ed = prob_wv[b, i_wn, ed, 1]
            p_wv1.append([p_st, p_ed])
        p_wv.append(p_wv1)

    return p_wv

def generate_inputs_s2s(tokenizer, nlu1_tt, hds1, sql_vocab1):
    """
    [CLS] sql_vocab [SEP] question [SEP] headers
    To make sql_vocab in a fixed position.
    """

    tokens = []
    segment_ids = []

    tokens.append("[CLS]")


    # sql_vocab
    i_sql_vocab = []
    # for doc
    for i, sql_vocab11 in enumerate(sql_vocab1):
        i_st_sql = len(tokens)
        sub_tok = tokenizer.tokenize(sql_vocab11)
        tokens += sub_tok
        i_ed_sql = len(tokens)
        i_sql_vocab.append((i_st_sql, i_ed_sql))
        segment_ids += [1] * len(sub_tok)
        if i < len(sql_vocab1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(sql_vocab1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError


    # question
    i_st_nlu = len(tokens)  # to use it later

    segment_ids.append(0)
    for token in nlu1_tt:
        tokens.append(token)
        segment_ids.append(0)
    i_ed_nlu = len(tokens)
    tokens.append("[SEP]")
    segment_ids.append(0)
    i_nlu = (i_st_nlu, i_ed_nlu)


    # headers
    i_hds = []
    # for doc
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1)-1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError


    return tokens, segment_ids, i_sql_vocab, i_nlu, i_hds


def sort_pr_wc(pr_wc, g_wc):
    """
    Input: list
    pr_wc = [B, n_conds]
    g_wc = [B, n_conds]


    Return: list
    pr_wc_sorted = [B, n_conds]
    """
    pr_wc_sorted = []
    for b, pr_wc1 in enumerate(pr_wc):
        g_wc1 = g_wc[b]
        pr_wc1_sorted = []

        if set(g_wc1) == set(pr_wc1):
            pr_wc1_sorted = deepcopy(g_wc1)
        else:
            # no sorting when g_wc1 and pr_wc1 are different.
            pr_wc1_sorted = deepcopy(pr_wc1)

        pr_wc_sorted.append(pr_wc1_sorted)
    return pr_wc_sorted

