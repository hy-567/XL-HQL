import json
import os
import re

import matplotlib.pyplot as plt

path_wikisql = './data_and_model'
path_sql = os.path.join(path_wikisql, 'dataset_hql.json')
DATA=[]
with open(path_sql) as inf:
    for idx, line in enumerate(inf):
        hql_info_list = json.loads(line.strip())
        DATA.append(hql_info_list)


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
    if hql_sel['column'] == c:  ##与class同名的实体对象
        conds['sel_pro']=0
    elif hql_sel['column'] in pro:  ##别名as 以及 不是以完全的属性路径编写的
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
                        #print(where_pro)
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
                if len(hs1)>50:     ##嵌套属性
                    count+=1
                res.append(hs1)
        else:#父类属性
            if len(hs1) > 50:
                count += 1
            res.append(hs1)
    if count>5:
        flag=False
    return res,flag

def all_header(hs):
    res=[]
    for hs1 in hs:
        if hs1 in res:
            continue
        else:
            res.append(hs1)
    return res

hql_data=[]
##属性数量的频率
fre_pro = []
fre_info = []
fre_query = []

for i, d in enumerate(DATA):
    temp = {}

    if type:
        ##有参数
        temp['question'] = get_question_type(d)
    else:
        ##无参数
        temp['question'] = get_question(d)

    temp['question_token'] = get_question_token(temp['question'])
    fre_info.append(len(temp['question_token']))
    #info_map[len(temp['question_token'])] = info_map.get(len(temp['question_token']),0)+1

    ##class的拆分
    gt_class = d['splitHql']['from']
    c_t = re.split("\.", gt_class)
    if len(c_t) % 2 == 0:
        a1 = '.'.join(c_t[x] for x in range(int((len(c_t) + 1) / 2)))
        a2 = '.'.join(c_t[x] for x in range(int((len(c_t) + 1) / 2), len(c_t)))
        if a1 == a2:
            gt_class = a1

    temp['class'] = gt_class
    t = d['column']
    # t.insert(0, gt_class)

    temp['header'] = all_header(t)
    fre_pro.append(len(temp['header']))
    #pro_map[len(temp['header'])] = pro_map.get(len(temp['header']), 0) + 1

    #temp['hql'] = get_hql(d, temp['header'], temp['class'])
    fre_query.append(len(d['cleanedHql']))
    #query_map[len(d['cleanedHql'])] = query_map.get(len(d['cleanedHql']), 0) + 1

##制图

# 生成一组随机数据

# 绘制直方图
plt.hist(fre_info, bins=50, range=(1,100),density=False,
         histtype='bar',
         edgecolor='#6B8ABC')
# plt.title('Histogram')
plt.xlabel('Information lengths')
plt.ylabel('Frequency')
plt.grid(color = 'w',  linewidth =2.0 )

ax1=plt.gca()
ax1.xaxis.grid(color='#E9E9F1',linewidth=0.5)
ax1.yaxis.grid(color='#E9E9F1',linewidth=0.5)
ax1.patch.set_facecolor("#E9E9F1")    # 设置 ax1 区域背景颜色
ax1.patch.set_alpha(0.5)
plt.savefig('Info.pdf')
plt.show()

print(123)





