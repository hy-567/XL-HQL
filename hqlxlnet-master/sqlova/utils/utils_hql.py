import os, json
from copy import deepcopy

from PyDeepLX import PyDeepLX
from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import generate_perm_inv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data -----------------------------------------------------------------------------------------------
def get_question(d):
    s=[]
    if d['HQLMethodComment'] != '':
        s.append('The MethodComment is')
        s.append(d['HQLMethodComment'].replace('\r','').replace('\n','').replace('\t',''))
    if d['HQLMethod'] != '':
        s.append('The MethodName is')
        s.append(d['HQLMethod'])

    if len(d['HQLMethodParName'])>0:
        s.append('The MethodParameterName is')
        s.append(' '.join(list(map(lambda x:x,d['HQLMethodParName']))))

    if len(d['calledInMethod'])>0:
        s.append('The contain of CallContextMethodName are')
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
    res=' '.join(s)
    return res


#使用参数
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

def get_question_token(identifiter):
    result = []
    for word in re.split(r':| |,|;|\{|\}|\<|\>|\*|@', identifiter, flags=0):
        if word:
            result.append(word)

    return result

def getNumofCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2+1)] for j in range(lstr1+1)]
    maxNum = 0
    p = 0
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


def data_merge(path_hql,toy_model,toy_size,type,rule):
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
            #count+=1
            continue
        if type:
            temp['question'] = get_question_type(d)
        else:
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
            temp['header']= all_header(t)
        else:
            temp['header'] = clean_hader(t, gt_class)

        temp['hql']= get_hql(d,temp['header'],temp['class'])
        if temp['hql']['where'][0]>4:
            #count+=1
            continue
        hql_data.append(temp)

    print(count)
    del DATA
    return hql_data

def data_corss_merge(path_hql,toy_model,toy_size,type,rule):
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

        if type:
            ##有参数
            temp['question'] = get_question_type(d)
        else:
            ##无参数
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


def get_loader_wikisql(data_train, test_dev, bS, shuffle_train=True, shuffle_test=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=0,
        collate_fn=lambda x:x # now dictionary values are not merged!

    )

    test_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=test_dev,
        shuffle=shuffle_test,
        num_workers=0,
        collate_fn=lambda x:x, # now dictionary values are not merged!
        drop_last=True
    )

    return train_loader, test_loader


##当是类的时候就为该类。若是类. 则为属性 否则直接添加
def clean_hader(hs,c):
    res=[]
    for hs1 in hs:
        if hs1==c:
            res.append(hs1)

        elif hs1.find('.')!=-1:
            if hs1.startswith(c+'.'):
                s = re.split(c + '.', hs1, 1)[-1]
                if s in res:
                    continue
                else:
                    res.append(s)
            else:
                res.append(hs1)
        else:#父类属性
            res.append(hs1)
    return res


def all_header(hs):
    res = []
    for hs1 in hs:
        if hs1 in res:
            continue
        else:
            res.append(hs1)
    return res

def get_fields(t1s):
    nlu, nlu_t, sql_i, hs,cs = [], [], [], [],[]
    for t1 in t1s:
        nlu1=t1['question']
        nlu_t1=t1['question_token']
        sql_i1=t1['hql']
        hs1=t1['header']
        g_cs = t1['class']
        ##header的清洗 由于类.pro的性质 并且由于只关注的任务是预测最后一个点后面的属性

        nlu.append(nlu1)
        nlu_t.append(nlu_t1)
        sql_i.append(sql_i1)
        hs.append(hs1)
        cs.append(g_cs)

    return nlu, nlu_t, sql_i, hs,cs

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
    size=max(l_hs)
    if size<=4:
        size=4
    wenc_hs = wenc_hpu.new_zeros(len(l_hs), size, hS)
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


    return tokens, segment_ids, i_nlu, i_hds

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


def get_bert_output(model_bert, tokenizer, nlu_t, hds, max_seq_length,x_model,x_tokenizer):
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

    doc_tokens = []
    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):

        hds1 = hds[b]
        l_hs.append(len(hds1))


        # 1. 2nd tokenization using WordPiece
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).  子令牌在一级令牌中的位置
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.    ##第i个1级临牌在所有令牌中的位置
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
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_inputs(tokenizer, nlu_tt1, hds1,max_seq_length)
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
           l_n, l_hpu, l_hs, nlu_tt, t_to_tt_idx, tt_to_t_idx


def get_hql_wemb_n(i_nlu, l_n, hS, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):
        # [B, max_len, dim]
        # Fill zero for non-exist part.
        i_nlu1 = i_nlu[b]
        wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), 0:hS] = all_encoder_layer[b, i_nlu1[0]:i_nlu1[1], :]
    return wemb_n

def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
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

def generate_gpt2_inputs(tokenizer, nlu1_tok, hds1):
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
        if i < len(hds1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, segment_ids, i_nlu, i_hds

def generate_albert_inputs(tokenizer, nlu1_tok, hds1):
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
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        segment_ids += [1] * len(sub_tok)
        if i < len(hds1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(0)
        elif i == len(hds1) - 1:
            tokens.append("[SEP]")
            segment_ids.append(1)
        else:
            raise EnvironmentError


    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, segment_ids, i_nlu, i_hds

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
    if l_hpu_max<=4:
        l_hpu_max=4

    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0]), 0:hS] = all_encoder_layer[b, i_hds11[0]:i_hds11[1],:]

    return wemb_h

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

def get_albert_output( nlu_t, hds,x_model,x_tokenizer):
    """
    Transformer-xl模型采用的是Auto 字级拆分 因此对于有些合成词需要自己拆分(驼峰命名之类)
    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []  ##总Mask

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_hds = []

    nlu_tt = []

    batch_len = 0

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
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_albert_inputs(x_tokenizer, nlu_tt1, hds1)
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
    for i,t in enumerate(input_ids):
        while len(t) < batch_len:
            input_ids[i].append(0)
            input_mask[i].append(0)
            segment_ids[i].append(0)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)


    all_encoder_layer = x_model( input_ids=all_input_ids, attention_mask=all_input_mask,token_type_ids =all_segment_ids)
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer,  tokens, i_nlu, i_hds, l_n, l_hpu, l_hs, nlu_tt

def get_GPT2_output( nlu_t, hds,x_model,x_tokenizer):
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
        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_gpt2_inputs(x_tokenizer, nlu_tt1, hds1)
        input_ids1 = x_tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)

        if len(input_ids1)>batch_len:
            batch_len=len(input_ids1)

    for i,t in enumerate(input_ids):
        while len(t) < batch_len:
            input_ids[i].append(0)
            input_mask[i].append(0)
            segment_ids[i].append(0)
    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    all_encoder_layer = x_model( input_ids=all_input_ids,attention_mask=all_input_mask,token_type_ids=all_segment_ids)

    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer,  tokens, i_nlu, i_hds, l_n, l_hpu, l_hs, nlu_tt

def get_Xlnet_output( nlu_t, hds,x_model,x_tokenizer):
    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

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
        input_ids1 = x_tokenizer.convert_tokens_to_ids(tokens1)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_hds.append(i_hds1)

        if len(input_ids1)>batch_len:
            batch_len=len(input_ids1)

    for i,t in enumerate(input_ids):
        while len(t) < batch_len:
            input_ids[i].append(0)
            input_mask[i].append(0)
            segment_ids[i].append(0)
    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(segment_ids, dtype=torch.long).to(device)

    all_encoder_layer = x_model(all_input_ids, all_input_mask,token_type_ids=all_segment_ids)
    l_hpu = gen_l_hpu(i_hds)

    return all_encoder_layer,  tokens, i_nlu, i_hds, l_n, l_hpu, l_hs, nlu_tt


def get_wemb_bert(model_type,bert_config, nlu_t, hds,x_model,x_tokenizer, num_out_layers_n=1):
    if model_type == 'xlnet':
        encoder_layer, x_tokens, x_i_nlu, x_i_hds, \
        x_l_n, x_l_hpu, x_l_hs, x_nlu_tt,= get_Xlnet_output(nlu_t, hds, x_model,x_tokenizer)
    elif model_type == 'albert':
        encoder_layer, x_tokens, x_i_nlu, x_i_hds, \
            x_l_n, x_l_hpu, x_l_hs, x_nlu_tt, = get_albert_output(nlu_t, hds, x_model, x_tokenizer)
    elif model_type == 'gpt2':
        encoder_layer, x_tokens, x_i_nlu, x_i_hds, \
            x_l_n, x_l_hpu, x_l_hs, x_nlu_tt, = get_GPT2_output(nlu_t, hds, x_model, x_tokenizer)

    wemb_n = get_hql_wemb_n(x_i_nlu, x_l_n, bert_config.hidden_size, encoder_layer[0], num_out_layers_n)

    wemb_h = get_hql_wemb_h(x_i_hds,x_l_hpu,x_l_hs,bert_config.hidden_size,encoder_layer[0],num_out_layers_n)
    return wemb_n, wemb_h, x_l_n, x_l_hpu, x_l_hs,x_nlu_tt

def pred_sc(s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc = []
    for s_sc1 in s_sc:
        pr_sc.append(s_sc1.argmax().item())

    return pr_sc


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


def find_sub_list(sl, l):
    # from stack overflow.
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results

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



def get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, g_sql_i, pr_sql_i,mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sc = get_cnt_sc_list(g_sc, pr_sc)
    cnt_sa = get_cnt_sc_list(g_sa, pr_sa)
    cnt_wn = get_cnt_sc_list(g_wn, pr_wn)
    cnt_wc = get_cnt_wc_list(g_wc, pr_wc)
    cnt_wo = get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)

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



def sort_and_generate_pr_w(pr_sql_i):
    pr_wc = []
    pr_wo = []

    for b, pr_sql_i1 in enumerate(pr_sql_i):
        conds1 = pr_sql_i1["conds"]
        pr_wc1 = []
        pr_wo1 = []


        # Generate
        for i_wn, conds11 in enumerate(conds1):
            pr_wc1.append( conds11[0])
            pr_wo1.append( conds11[1])


        # sort based on pr_wc1
        idx = argsort(pr_wc1)
        pr_wc1 = array(pr_wc1)[idx].tolist()
        pr_wo1 = array(pr_wo1)[idx].tolist()


        conds1_sorted = []
        for i, idx1 in enumerate(idx):
            conds1_sorted.append( conds1[idx1] )


        pr_wc.append(pr_wc1)
        pr_wo.append(pr_wo1)


        pr_sql_i1['conds'] = conds1_sorted

    return pr_wc, pr_wo, pr_sql_i

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

            pr_wc1_sorted = deepcopy(pr_wc1)

        pr_wc_sorted.append(pr_wc1_sorted)
    return pr_wc_sorted

