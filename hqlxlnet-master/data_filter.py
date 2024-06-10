import json
import os
import re
from PyDeepLX import PyDeepLX

path_wikisql = './'
path_sql = os.path.join(path_wikisql, 'dataset.json')

DATA=[]
with open(path_sql,encoding='gbk') as inf:
    for idx, line in enumerate(inf):
        hql_info_list = json.loads(line.strip())
        DATA.append(hql_info_list)

hql_data=[]

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

for i, d in enumerate(DATA):
    if contains_non_english(d['HQLMethodComment']):
        #print(d['HQLMethodComment'])
        d['HQLMethodComment'] = translate(d['HQLMethodComment'])
    hql_data.append(d)

filename = "dataset_hql.json"
with open(filename, 'w') as file_obj:
    for item in hql_data:
        json.dump(item, file_obj)
        file_obj.write('\n')


