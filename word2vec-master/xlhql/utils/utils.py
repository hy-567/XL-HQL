# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
import os, json
import random as python_random
from matplotlib.pylab import *


##cameCase命名
def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def snake_case_split(identifiter):
    result = []
    for line in identifiter:
        for word in re.split(r'_|\.| |,|;|\)|\(|\{|\}|\<|\>|:|\*|@', line, flags=0):
            if word:
                result.append(word.lower())
    return result

def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = zeros(len(perm), dtype=int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv


def ensure_dir(my_path):
    """ Generate directory if not exists
    """
    if not os.path.exists(my_path):
        os.makedirs(my_path)


def topk_multi_dim(tensor, n_topk=1, batch_exist=True):

    if batch_exist:
        idxs = []
        for b, tensor1 in enumerate(tensor):
            idxs1 = []
            tensor1_1d = tensor1.reshape(-1)
            values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
            idxs_list = unravel_index(idxs_1d.cpu().numpy(), tensor1.shape)
            # (dim0, dim1, dim2, ...)

            # reconstruct
            for i_beam in range(n_topk):
                idxs11 = []
                for idxs_list1 in idxs_list:
                    idxs11.append(idxs_list1[i_beam])
                idxs1.append(idxs11)
            idxs.append(idxs1)

    else:
        tensor1 = tensor
        idxs1 = []
        tensor1_1d = tensor1.reshape(-1)
        values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
        idxs_list = unravel_index(idxs_1d.numpy(), tensor1.shape)
        # (dim0, dim1, dim2, ...)

        # reconstruct
        for i_beam in range(n_topk):
            idxs11 = []
            for idxs_list1 in idxs_list:
                idxs11.append(idxs_list1[i_beam])
            idxs1.append(idxs11)
        idxs = idxs1
    return idxs


def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, int64): return int(o)
    raise TypeError


def load_jsonl(path_file, toy_data=False, toy_size=4, shuffle=False, seed=1):
    data = []

    with open(path_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size and (not shuffle):
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print(
            f"If the toy-data is used, the whole data loaded first and then shuffled before get the first {toy_size} data")

        python_random.Random(seed).shuffle(data)  # fixed
        data = data[:toy_size]

    return data

##候选属性名称的分割
def properties_name(data):
    properties_token=[]
    pros=data['column']
    pro=data['splitHql']['from']
    for por in pros:
        properties_token.append(snake_case_split(camel_case_split(por)))
    properties_token.insert(0,snake_case_split(camel_case_split(pro)))
    pros.insert(0,pro)

    return properties_token

##候选属性名称的类型分割
def properties_name_par(data):
    pro_types=data['columnType']
    pro_type_token=[]
    for idy, token in enumerate(pro_types):
        s = []
        if len(token) == 0:
            continue
        else:
            if '<' in token:
                token=token.replace('<', ' ').strip('>')
            for x in snake_case_split(camel_case_split(token)):
                s.append(x)
        pro_type_token.append(s)

    return pro_type_token
##候选类的拆分
def from_class(data):
    return snake_case_split(camel_case_split(data))
##获得注释token
def getcomment_token(data):
    hql_comment = data['HQLMethodComment']
    return snake_case_split(camel_case_split(hql_comment)) if len(snake_case_split(camel_case_split(hql_comment)))!=0 else []

##获得query_token
def getquery_token(data):
    hql_query=data['cleanedHql']
    mathches=re.split(' ',hql_query)
    return mathches

##获得query中的select以及where的限制条件
def get_query(data):
    hql_query=data['splitHql']
    hql_sel=hql_query.get('select')
    sel_pro=hql_sel.get('column')
    sel_agg=hql_sel.get('agg','None')
    gt_sel= []
    gt_where=[]
    agg = {'NONE':0,'COUNT':1, 'AVG':2, 'MIN':3, 'MAX':4, 'SUM':5}
    ##操作符限制  等于 大于 小于
    COND_OPS = {'UNKNOWN':0 ,'=': 1, '!=': 2, 'IS NULL': 3, 'IS NOT NULL': 4, '<': 5, '>': 6, 'IN':7, 'NOT IN': 8, 'LIKE': 9}
    gt_sel.append(agg.get(sel_agg.upper()))
    gt_sel.append(sel_pro)

    gt_class=hql_query.get('from')
    ##class的拆分
    temp = re.split("\.", gt_class)
    if len(temp)%2==0:
        a1 = '.'.join(temp[x] for x in range(int((len(temp) + 1) / 2)))
        a2 = '.'.join(temp[x] for x in range(int((len(temp) + 1) / 2), len(temp)))
        if a1==a2:
            gt_class=a1

    hql_where=hql_query.get('where')
    gt_value = []
    if len(hql_where)>0:
        for dic_where in hql_where:
            temp=[]
            where_op=dic_where.get('op','UNKNOWN')
            if where_op=='>=':
                where_op='>'
            elif where_op=='<=':
                where_op='<'
            elif where_op.upper()=='NOT LIKE':
                where_op='LIKE'
                print('NOT LIKE被替换')
            temp.append(COND_OPS.get(where_op.upper(),0))

            where_pro=dic_where.get('column')
            temp.append(where_pro)
            gt_value.append(temp)
        where_num=len(hql_where)
    else:
        temp = []
        where_pro='None'
        temp.append('None')
        temp.append(where_pro)
        gt_value.extend(temp)
        where_num=0

    gt_where.append(where_num)
    gt_where.append(gt_value)
    return gt_sel,gt_where,gt_class

##获取方法名的token
def target_name(data):
    method_name=data['HQLMethod']
    hql_method_name_token=snake_case_split(camel_case_split(method_name))
    return hql_method_name_token

##获取方法参数以及参数类型
def target_par(data):
    hql_method_par=data['HQLMethodParName']
    hql_method_par_token = []
    for data_column in hql_method_par:
        s = []
        ##令牌判空操作
        if not len(data_column):
            s.append([])
            continue
        else:
            ## 驼峰和_分割
            s=snake_case_split(camel_case_split(data_column))
        hql_method_par_token.append(s)
    return hql_method_par_token

##获取方法参数的类型
def target_par_type(data):
    ##方法参数类型分割
    par_type_token=data['HQLMethodParType']
    hql_method_par_type_token = []
    for idy, token in enumerate(par_type_token):
        s=[]
        if len(token)==0:
            continue
        else:
            for x in snake_case_split(camel_case_split(token)):
                s.append(x.lower())
        hql_method_par_type_token.append(s)
    return hql_method_par_type_token

##call context的 方法名 参数 参数类型的token
def call_method_name(data):
    call_methods=data['calledInMethod']
    hql_context_method_token = []
    for data_context in call_methods:
        s=[]
        if len(data_context)!=0:
            s=snake_case_split(camel_case_split(data_context))
        hql_context_method_token.append(s)
    return  hql_context_method_token

def call_method_par(data):
    call_method_pars=data['calledInPar']
    hql_context_token = []
    for data_column in call_method_pars:
        s = []
        ##令牌判空操作
        if not len(data_column):
            hql_context_token.append(s)
            continue
        else:
            ## 驼峰和_分割
            for token in data_column:
                s=snake_case_split(camel_case_split(token))
            hql_context_token.append(s)
    return hql_context_token

def call_method_par_type(data):
    call_method_par_types=data['calledInParType']
    ##context_par_type令牌分割
    call_method_par_type_token = []
    for column_type in call_method_par_types:
        s = []
        if not len(column_type):
            call_method_par_type_token.append(s)
            continue
        else:
            for token in column_type:
                if '<' in token:
                    token=token.replace('<', ' ').strip('>')
                s=snake_case_split(camel_case_split(token))
        call_method_par_type_token.append(s)
    return call_method_par_type_token