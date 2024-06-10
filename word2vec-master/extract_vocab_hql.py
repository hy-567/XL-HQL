from xlhql.utils.utils import *
import numpy as np
from gensim.models import word2vec

'''
 **cleanedHql**: Rewritten HQL query according to the template (not used by HQLgen)
 **splitHql**: HQL query split by clause
   **select**: **SELECT** clause
     **agg**: Aggregate function if exists
     **column**: Selected property (may be the whole class)
    **fullExpr**: Full **SELECT** expression
 **from**: Target class name
 **where**: **WHERE** clause
    **column**: Property of a condition
    **op**: Operator of a condition
    **value**: Placeholder, not predicted
    **fullExpr**: Full **WHERE** expression
HQLMethod**: Target method name
HQLMethodParName**: Parameter names of the target method
HQLMethodParType**: Parameter types of the target method
HQLMethodComment**: Comment of the target method if exists
calledInMethod**: Method names of the methods that call the target method (call context)   
calledInPar**: Parameter names in the call context
calledInParType**: Parameter types in the call context
column**: Candidate property names in the target class
columnType**: Types of the candidate properties
testFile**: Test project or not
projectName**: Project name
{'question': 'Which country is Jim Les from?',
 'query_tok': ['SELECT', 'nationality', 'WHERE', 'player', 'EQL', 'jim', 'les'], 
 'query_tok_space': [' ', ' ', ' ', ' ', ' ', ' ', ''], 
 'table_id': '1-11545282-12', 
 'question_tok_space': [' ', ' ', ' ', ' ', ' ', '', ''], 
 'sql': {'agg': 0, 'sel': 2, 
 'conds': [[0, 0, 'Jim Les']]}, 
 'phase': 1, 
 'query': 'SELECT nationality WHERE player EQL jim les', 
 'question_tok': ['which', 'country', 'is', 'jim', 'les', 'from', '?']},
'''
def get_question(d):
    s=[]
    if d['HQLMethodComment'] != '':
        s.append('The MethodComment is')
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
    res=' '.join(s)
    return res


def get_question_token(identifiter):
    result = []
    for s in re.split(r':| |,|;|\)|\(|\{|\}|\<|\>|\*|@|#', identifiter, flags=0):
        result.append(s.lower())

    return result

def load_dataset(sql_paths,use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    hql_data = []
    for SQL_PATH in sql_paths:
        print("Loading data from %s"%SQL_PATH)
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                hql_info_list = json.loads(line.strip())
                hql_data.append(hql_info_list)
    return hql_data

path=load_dataset("data_and_model/dataset_hql.json")
embs = [['<UNK>', '<BEG>', '<END>', '<\\m>', '<\\t>']]

for i in range(len(path)):
    pro_token=properties_name(path[i])
    pro_type_token=properties_name_par(path[i])
    com_token=getcomment_token(path[i])
    q_token=getquery_token(path[i])
    target_name_token=target_name(path[i])
    target_par_token=target_par(path[i])
    #target_par_type_token=target_par_type(path[i])
    call_name_token=call_method_name(path[i])
    call_par_token=call_method_par(path[i])
    #call_par_type_token=call_method_par_type(path[i])
    question_token=get_question_token(get_question(path[i]))


    ##处理嵌入
    embs[0].extend(q_token)
    if len(com_token)!=0:
        embs[0].extend(com_token)
    for token in pro_token:
        embs[0].extend(token)
    for token in pro_type_token:
        embs[0].extend(token)
    embs[0].extend(target_name_token)
    for p in target_par_token:
        if len(p)!=0:
            embs[0].extend(p)
    # for pr in target_par_type_token:
    #     if len(pr)!=0:
    #         embs[0].extend(pr)
    for pr in call_name_token:
        if len(pr)!=0:
            embs[0].extend(pr)
    for pr in call_par_token:
        if len(pr)!=0:
            embs[0].extend(pr)
    # for pr in call_par_type_token:
    #     if len(pr)!=0:
    #         embs[0].extend(pr)
    embs[0].extend(question_token)

model = word2vec.Word2Vec(embs,sg=0,hs=1, min_count=1, window=5,workers=4,alpha=0.001,vector_size=96,epochs=100)
model.build_vocab(embs)
model.save("data_and_model/hql_word2vec.model")