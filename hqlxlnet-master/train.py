import os, sys, argparse, re, json
import random as python_random

from sklearn.model_selection import KFold

from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from sqlova.model.nl2sql.xlhql_models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, GPT2Model
from transformers import GPT2Tokenizer, GPT2Config

from transformers import AutoTokenizer, AlbertModel
from transformers import AlbertConfig, AlbertModel

def construct_hyper_param(parser):
    parser.add_argument("--do_train", default=False, action='store_true')

    parser.add_argument("--trained", default=False, action='store_true')

    parser.add_argument('--tepoch', default=40, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        )

    parser.add_argument("--model_type", default='xlnet', type=str,
                        help="Type of model.")

    parser.add_argument('--project', default=False,action="store_true")
    parser.add_argument('--type', default=False, action='store_true')
    parser.add_argument('--rule', default=False, action='store_true')

    # 1.2 Pretrained Parameters
    parser.add_argument("--num_target_layers",default=1, type=int,)
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    args = parser.parse_args()


    print(f"BERT-type: {args.model_type}")

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.toy_model = False
    args.toy_size = 12

    return args


def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert


def  get_models(args, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = {'NONE', 'COUNT', 'AVG', 'MIN', 'MAX', 'SUM'}
    cond_ops = ['UNKNOWN', '=', '!=', 'IS NULL', 'IS NOT NULL', '<', '>', 'IN', 'NOT IN','LIKE']  # do not know why 'OP' required. Hence,

    if args.model_type=='xlnet':
        x_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        x_model = XLNetModel.from_pretrained('xlnet-base-cased')  ##用的是预训练模型是无法改变其参数的
        xl_config = x_model.config
    elif args.model_type=='albert':
        albert_base_configuration = AlbertConfig(
            hidden_size=768,
            num_attention_heads=12,
            max_position_embeddings=4096
        )
        x_model = AlbertModel(albert_base_configuration)
        x_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        xl_config = x_model.config
    elif args.model_type=='gpt2':
        GPT2_base_configuration = GPT2Config(
            n_positions=2048
        )
        x_model = GPT2Model(GPT2_base_configuration)
        x_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        xl_config = x_model.config

    x_model.to(device)
    args.iS = xl_config.hidden_size

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = HQL_XL(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')

        if args.model_type == 'xlnet':
            x_model.load_state_dict(res['xlnet_model'],False)
        elif args.model_type=='albert':
            x_model.load_state_dict(res['albert_model'])
        elif args.model_type=='gpt2':
            x_model.load_state_dict(res['gpt2_model'], False)
        x_model.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'],False)

    return model, x_model, x_tokenizer, xl_config

def mergeData(hql_datas):
    merge_data=[]
    for hql_data in hql_datas:
        temp=''
        temp+=hql_data['question']
        temp+=' [SEP] '
        hql_headers=hql_data['header']
        for h in hql_headers:
            temp+=' '+h+' [SEP] '
        temp+='[CLS]'
        merge_data.append(temp)
    return merge_data

def train(model_type,train_loader, model, opt, bert_config,x_model,x_tokenizer, num_target_layers, accumulate_gradients=1, st_pos=0, opt_bert=None):
    model.train()
    x_model.train()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_lx = 0  # of logical form acc

    count_agg_pr=[]
    count_agg_gt = []
    count_cond_pr=[]
    count_cond_gt=[]

    for iB, t in enumerate(train_loader):
        cnt += len(t)   ##batch_size

        if cnt < st_pos:
            continue
        # Get fields  获得token
        nlu, nlu_t, sql_i, hds,cs = get_fields(t)

        g_sc, g_sa, g_wn, g_wc, g_wo = get_g(sql_i)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, nlu_tt= get_wemb_bert(model_type,bert_config, nlu_t, hds,x_model,x_tokenizer, num_out_layers_n=num_target_layers)


        # score
        s_sc, s_sa, s_wn, s_wc, s_wo = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc)

        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, g_sc, g_sa, g_wn, g_wc, g_wo)

        # Calculate gradient
        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo)

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

        count_agg_pr.extend(pr_sa)
        count_agg_gt.extend(g_sa)

        count_cond_pr.extend(pr_wn)
        count_cond_gt.extend(g_wn)


    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_lx = cnt_lx / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_lx]
    return acc

def test(data_loader, model, model_bert, bert_config, tokenizer, num_target_layers,st_pos=0):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_lx = 0  # of logical form acc

    for iB, t in enumerate(data_loader):
        cnt += len(t)  ##batch_size

        if cnt < st_pos:
            continue

        nlu, nlu_t, sql_i, hds,cs = get_fields(t)
        g_sc, g_sa, g_wn, g_wc, g_wo = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        ##放入bert中做词嵌入
        wemb_n, wemb_h, l_n, l_hpu, l_hs, nlu_tt = get_wemb_bert(bert_config, nlu_t, hds,model_bert, tokenizer, num_out_layers_n=num_target_layers)


        # score
        s_sc, s_sa, s_wn, s_wc, s_wo = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                             g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc)

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

def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_lx = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f},acc_lx: {acc_lx:.3f}"
    )

def get_real_dataset(datasets):
    res = []
    for datas in datasets:
        for data in datas.values():
            res.extend(data)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_h = './data_and_model'
    path_wikisql = './data_and_model'
    BERT_PT_PATH = path_wikisql

    path_save_for_evaluation = './'

    ##交叉验证
    skf = KFold(n_splits=10, shuffle=True, random_state=42)

    ##merge_data
    if args.project:
        project_data = data_corss_merge(path_wikisql, args.toy_model, args.toy_size,type= args.type,rule=args.rule)
        hql_project = [{key: value} for key, value in project_data.items()]
        project_data = np.array(hql_project)
    else:
        hql_data=data_merge(path_wikisql,args.toy_model, args.toy_size,type= args.type,rule=args.rule)
        hql_np = np.array(hql_data)

    ## 4. Build & Load models
    if not args.trained:
        model, x_model, x_tokenizer, xl_config = get_models(args)
    else:
        # To start from the pre-trained models
        path_model_bert = './cross_project/xlnet_cross_best.pt'
        path_model = './cross_project/model_cross_best.pt'
        model, x_model, x_tokenizer, xl_config = get_models(args, trained=True, path_model_bert=path_model_bert, path_model=path_model)

    ## 5. Get optimizers
    if args.do_train:
        opt, opt_bert = get_opt(model, x_model, args.fine_tune)
        acc_lx_t_best = -1
        epoch_best = -1

    if args.project:
        for epoch in range(args.tepoch):
            print('Epoch %d @ %s' % (epoch + 1, datetime.datetime.now()))
            test_ave_acc=0
            acc_temp=[]
            for train_index, test_index in skf.split(hql_project):
                train_data, test_data = project_data[train_index].tolist(), project_data[test_index].tolist()
                train_data = get_real_dataset(train_data)
                test_data = get_real_dataset(test_data)
                train_loader,test_loader= get_loader_wikisql(train_data, test_data,args.bS, shuffle_train=True, shuffle_test=True)
                acc_train = train(args.model_type,train_loader,
                                             model,
                                             opt,
                                             xl_config,
                                             x_model,
                                             x_tokenizer,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0)


                # check DEV
                with torch.no_grad():
                    acc_dev = test(data_loader=test_loader, model=model, model_bert=x_model, bert_config=xl_config, tokenizer=x_tokenizer,num_target_layers=args.num_target_layers,st_pos=0,EG=args.EG)
                    acc_temp.append(acc_dev[-1])

                print_result(epoch+1, acc_train, 'train')
                print_result(epoch+1, acc_dev, 'dev')

            test_ave_acc=np.mean(acc_temp)


            if test_ave_acc>acc_lx_t_best:
                acc_lx_t_best=test_ave_acc
                epoch_best = epoch+1
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join('.', 'model_best.pt'))

                state = {'xlnet_model': x_model.state_dict()}
                torch.save(state, os.path.join('.', 'xlnet_best.pt'))

            print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
    else:
        for epoch in range(args.tepoch):
            print('Epoch %d @ %s' % (epoch + 1, datetime.datetime.now()))
            test_ave_acc=0
            acc_temp=[]
            for train_index, test_index in skf.split(hql_data):
                train_data, test_data = hql_np[train_index].tolist(), hql_np[test_index].tolist()
                train_loader,test_loader= get_loader_wikisql(train_data, test_data,args.bS, shuffle_train=True, shuffle_test=True)
                acc_train = train(args.model_type,train_loader,
                                             model,
                                             opt,
                                             xl_config,
                                             x_model,
                                             x_tokenizer,
                                             args.num_target_layers,
                                             args.accumulate_gradients,
                                             opt_bert=opt_bert,
                                             st_pos=0)


                # check DEV
                with torch.no_grad():
                    acc_dev = test(data_loader=test_loader, model=model, model_bert=x_model, bert_config=xl_config, tokenizer=x_tokenizer,num_target_layers=args.num_target_layers,st_pos=0)
                    acc_temp.append(acc_dev[-1])

                print_result(epoch+1, acc_train, 'train')
                print_result(epoch+1, acc_dev, 'dev')

            test_ave_acc=np.mean(acc_temp)


            if test_ave_acc>acc_lx_t_best:
                acc_lx_t_best=test_ave_acc
                epoch_best = epoch+1
                state = {'model': model.state_dict()}
                torch.save(state, os.path.join('.', 'model_best.pt'))

                state = {'xlnet_model': x_model.state_dict()}
                torch.save(state, os.path.join('.', 'xlnet_best.pt'))

            print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")