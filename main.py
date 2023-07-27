import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from scipy import sparse
import models
import data
import os
import pandas as pd
parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='amazon',
                    help='select dataset')
parser.add_argument('--stage', type=str, default='backbone',
                    help='training stage:backbone,uncertain')
parser.add_argument('--model', type=str, default='VAE',
                    help='backbone model:VAE,MF,LGCN')
parser.add_argument('--activate_function', type=str, default='linear',
                    help='activate function in uncertainty estimator')
parser.add_argument('--device', type=int, default=0,
                    help='cuda id')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--alpha', type=float, default=1,
                    help='hyper-parameter in paper')
parser.add_argument('--beta', type=float, default=1e-2,
                    help='hyper-parameter in paper')
parser.add_argument('--gamma', type=float, default=1e-3,
                    help="hyper-parameter in paper")
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=10000,
                    help='upper epoch limit')
parser.add_argument('--neg_sample_u', type=float, default=1.0,
                    help='neg sample ')
parser.add_argument('--user_uncertain', action="store_true",
                    help='whether use user_uncertain')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--save', type=str, default='test.pkl',
                    help='path to save the final model')
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.set_device(args.device)
pro_dir = os.path.join(args.data, 'pro_sg')
train_df=pd.read_csv(os.path.join(pro_dir,'train.csv'))
test_df=pd.read_csv(os.path.join(pro_dir,'test_te.csv'))
uratio=train_df.uid.unique().shape[0]/test_df.uid.unique().shape[0]

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda" if args.cuda else "cpu")
loader = data.DataLoader(args.data)
n_items = loader.load_n_items()
train_data = loader.load_data('train')
# vad_data_tr, vad_data_te = loader.load_data('validation')
test_data_tr, test_data_te = loader.load_data('test')
p_dims=[512,1024,n_items]

num_user,num_item=train_data.shape[0],train_data.shape[1]
print(num_user,num_item)
print(test_data_tr.shape,test_data_te.shape)
if args.model=='VAE':
    model = models.MultiVAE_Linear(p_dims,dropout=0,activate_function=args.activate_function,user_uncertain=args.user_uncertain,n_user=num_user,stage=args.stage).to(device)
elif args.model=='LGCN':
    model = models.LGCN_Linear(user_item_net=train_data, num_item=num_item, n_layers=3,
                               num_user=num_user,dataset=args.data,user_uncertain=args.user_uncertain,stage=args.stage ).to(device)
    model.get_adj()
elif args.model=='MF':
    model = models.LGCN_Linear(user_item_net=train_data, num_item=num_item, n_layers=0,
                               num_user=num_user,dataset=args.data,user_uncertain=args.user_uncertain,stage=args.stage).to(device)
#################load save
if args.stage=='backbone':
    pass
else:
    save_model=torch.load(args.data+'_'+args.model+'.pkl',map_location='cpu')
    model_dict =  model.state_dict()

    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}

    print(state_dict.keys())

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
model.init_c()

N = train_data.shape[0]
idxlist = list(range(N))
idxlist_train = list(range(N))
optimizer= optim.Adam(model.parameters(), lr=args.lr)
pop=np.array(train_data.sum(0))[0]
idcg_list=torch.zeros(1000)
for i in range(1,1000):
    idcg_list[i]+=idcg_list[i-1]+1/np.log2(i+1)
if args.data=='amazon':
    sep_=39
elif args.data=='adressa' or args.data=='globo' or args.data=='adressa_macr':
    sep_=500
elif args.data=='gowalla':
    sep_=34
else:
    sep_=56
def evaluate_fast_all(model,topK=20,uratio=1.0):
    test_batch = 256
    top = topK
    score_recall = []
    uncertain_recall = []
    score_ndcg = []
    uncertain_ndcg = []
    model.eval()
    for s_index in range(0, N, test_batch):
        e_index = min(N, s_index + test_batch)
        data = train_data[idxlist[s_index:e_index]]
        data = naive_sparse2tensor(data)
        data = data.to(device)
        recon_batch, c = model(data, idxlist[s_index:e_index])
        ground_truth = test_data_te[s_index:e_index].toarray()
        recon_batch -= 100 * (data)
        c -= 100 * (data)
        rank_score = torch.topk(recon_batch, top)[-1]
        rank_uncertain = torch.topk(c, top)[-1]
        hit_score_r = torch.zeros_like(recon_batch)
        hit_uncertain_r = torch.zeros_like(c)
        hit_score_n = torch.zeros_like(recon_batch)
        hit_uncertain_n = torch.zeros_like(c)
        hit_score_r[torch.arange(e_index - s_index).unsqueeze(-1), rank_score] = 1
        hit_score_n[torch.arange(e_index - s_index).unsqueeze(-1), rank_score] = 1 / torch.log2(
            2 + torch.arange(top).cuda().float())
        hit_uncertain_r[torch.arange(e_index - s_index).unsqueeze(-1), rank_uncertain] = 1
        hit_uncertain_n[torch.arange(e_index - s_index).unsqueeze(-1), rank_uncertain] = 1 / torch.log2(
            2 + torch.arange(top).cuda().float())
        hit_score_r = hit_score_r.cpu() * ground_truth
        hit_score_n = hit_score_n.cpu() * ground_truth
        hit_uncertain_r = hit_uncertain_r.cpu() * ground_truth
        hit_uncertain_n = hit_uncertain_n.cpu() * ground_truth
        ground_hit = ground_truth.sum(1)
        user_hit_score_r, user_hit_uncertain_r = hit_score_r.sum(1) / (ground_hit + 1e-5), hit_uncertain_r.sum(1) / (
                    ground_hit + 1e-5)
        ground_hit[ground_hit > top] = top
        idcg = idcg_list[ground_hit]

        score_recall.append(user_hit_score_r)
        uncertain_recall.append(user_hit_uncertain_r)
        score_ndcg.append(hit_score_n.sum(1) / (idcg + 1e-5))
        uncertain_ndcg.append(hit_uncertain_n.sum(1) / (idcg + 1e-5))
    return torch.cat(score_recall).mean()*uratio, torch.cat(uncertain_recall).mean()*uratio, torch.cat(score_ndcg).mean()*uratio, torch.cat(
        uncertain_ndcg).mean()*uratio


def evaluate_fast_t1(model,topK=20,sep=39,uratio=1.0):
    test_batch = 256
    top = topK
    score_recall = []
    uncertain_recall = []
    score_ndcg = []
    uncertain_ndcg = []
    model.eval()
    for s_index in range(0, N, test_batch):
        e_index = min(N, s_index + test_batch)
        data = train_data[idxlist[s_index:e_index]]
        data = naive_sparse2tensor(data)
        data = data.to(device)
        recon_batch, c = model(data, idxlist[s_index:e_index])

        ground_truth = test_data_te[s_index:e_index].toarray() * (pop <=sep).astype(int)

        recon_batch -= 100 * (data)
        c -= 100 * (data)
        rank_score = torch.topk(recon_batch, top)[-1]
        rank_uncertain = torch.topk(c, top)[-1]
        hit_score_r = torch.zeros_like(recon_batch)
        hit_uncertain_r = torch.zeros_like(c)
        hit_score_n = torch.zeros_like(recon_batch)
        hit_uncertain_n = torch.zeros_like(c)
        hit_score_r[torch.arange(e_index - s_index).unsqueeze(-1), rank_score] = 1
        hit_score_n[torch.arange(e_index - s_index).unsqueeze(-1), rank_score] = 1 / torch.log2(
            2 + torch.arange(top).cuda().float())
        hit_uncertain_r[torch.arange(e_index - s_index).unsqueeze(-1), rank_uncertain] = 1
        hit_uncertain_n[torch.arange(e_index - s_index).unsqueeze(-1), rank_uncertain] = 1 / torch.log2(
            2 + torch.arange(top).cuda().float())
        hit_score_r = hit_score_r.cpu() * ground_truth
        hit_score_n = hit_score_n.cpu() * ground_truth
        hit_uncertain_r = hit_uncertain_r.cpu() * ground_truth
        hit_uncertain_n = hit_uncertain_n.cpu() * ground_truth
        ground_hit = ground_truth.sum(1)
        user_hit_score_r, user_hit_uncertain_r = hit_score_r.sum(1) / (ground_hit + 1e-5), hit_uncertain_r.sum(1) / (
                    ground_hit + 1e-5)
        ground_hit[ground_hit > top] = top
        idcg = idcg_list[ground_hit]

        score_recall.append(user_hit_score_r)
        uncertain_recall.append(user_hit_uncertain_r)
        score_ndcg.append(hit_score_n.sum(1) / (idcg + 1e-5))
        uncertain_ndcg.append(hit_uncertain_n.sum(1) / (idcg + 1e-5))
    return torch.cat(score_recall).mean()*uratio, torch.cat(uncertain_recall).mean()*uratio, torch.cat(score_ndcg).mean()*uratio, torch.cat(
        uncertain_ndcg).mean()*uratio


def evaluate_fast_t2(model,topK=20,sep=39,uratio=1.0):
    test_batch = 256
    top = topK
    score_recall = []
    uncertain_recall = []
    score_ndcg = []
    uncertain_ndcg = []
    model.eval()
    for s_index in range(0, N, test_batch):
        e_index = min(N, s_index + test_batch)
        data = train_data[idxlist[s_index:e_index]]
        data = naive_sparse2tensor(data)
        data = data.to(device)
        recon_batch, c = model(data, idxlist[s_index:e_index])
        #         recon_batch,c=model(data)
        #         print(torch.max(recon_batch-c),torch.min(recon_batch-c))
        ground_truth = test_data_te[s_index:e_index].toarray() * (pop <= sep).astype(int)
        #         c=torch.exp(c)*0.1
        #         c=0.2*recon_batch+0.8*c
        #         ground_truth=test_data_te[s_index:e_index].toarray()
        recon_batch -= 100 * (data + torch.Tensor((pop > sep).astype(float)).cuda())
        c -= 100 * (data + torch.Tensor((pop > sep).astype(float)).cuda())
        rank_score = torch.topk(recon_batch, top)[-1]
        rank_uncertain = torch.topk(c, top)[-1]
        hit_score_r = torch.zeros_like(recon_batch)
        hit_uncertain_r = torch.zeros_like(c)
        hit_score_n = torch.zeros_like(recon_batch)
        hit_uncertain_n = torch.zeros_like(c)
        hit_score_r[torch.arange(e_index - s_index).unsqueeze(-1), rank_score] = 1
        hit_score_n[torch.arange(e_index - s_index).unsqueeze(-1), rank_score] = 1 / torch.log2(
            2 + torch.arange(top).cuda().float())
        hit_uncertain_r[torch.arange(e_index - s_index).unsqueeze(-1), rank_uncertain] = 1
        hit_uncertain_n[torch.arange(e_index - s_index).unsqueeze(-1), rank_uncertain] = 1 / torch.log2(
            2 + torch.arange(top).cuda().float())
        hit_score_r = hit_score_r.cpu() * ground_truth
        hit_score_n = hit_score_n.cpu() * ground_truth
        hit_uncertain_r = hit_uncertain_r.cpu() * ground_truth
        hit_uncertain_n = hit_uncertain_n.cpu() * ground_truth
        ground_hit = ground_truth.sum(1)
        user_hit_score_r, user_hit_uncertain_r = hit_score_r.sum(1) / (ground_hit + 1e-5), hit_uncertain_r.sum(1) / (
                    ground_hit + 1e-5)
        ground_hit[ground_hit > top] = top
        idcg = idcg_list[ground_hit]

        score_recall.append(user_hit_score_r)
        uncertain_recall.append(user_hit_uncertain_r)
        score_ndcg.append(hit_score_n.sum(1) / (idcg + 1e-5))
        uncertain_ndcg.append(hit_uncertain_n.sum(1) / (idcg + 1e-5))
    return torch.cat(score_recall).mean()*uratio, torch.cat(uncertain_recall).mean()*uratio, torch.cat(score_ndcg).mean(), torch.cat(
        uncertain_ndcg).mean()*uratio
def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())
def sample_neg_new(data_csr,ratio=0.1):
    data = data_csr.toarray()
    n_item = np.zeros_like(data)
    l = int(data.shape[1] * ratio)
    neg_index = np.random.choice(np.arange(n_items), l, replace=False)
    n_item[:,neg_index] = 1
    return torch.Tensor(n_item)

def train_one_epoch_vae():
    np.random.shuffle(idxlist_train)
    model.train()
    stime = time.time()
    loss_r_list=[]
    loss_rate_list=[]
    c_list=[]
    for batch_idx,start_idx in enumerate(range(0,N,args.batch_size)):
        end_idx=min(start_idx + args.batch_size, N)
        optimizer.zero_grad()
        idx_batch=idxlist_train[start_idx:end_idx]
        data = train_data[idxlist_train[start_idx:end_idx]]

        if args.neg_sample_u==1:
            data = naive_sparse2tensor(data)
            n_data=1-data
            n_data=n_data.to(device)
        else:
            n_data = sample_neg_new(data, args.neg_sample_u)
            n_data=torch.Tensor(n_data).to(device)
            data = naive_sparse2tensor(data)
        data = data.to(device)
        recon_batch, c= model(data,idx_batch)
        if args.stage=='backbone':
            pos_loss = torch.mean(torch.sum(data * (recon_batch * data - 1) ** 2, -1))
            neg_loss = torch.mean(torch.sum(n_data * (recon_batch * n_data - 0) ** 2, -1))
            loss_r = (args.alpha * pos_loss + neg_loss)
        else:
            pos_loss=torch.mean(torch.sum(data*(recon_batch.detach()*data-1)**2/(torch.exp(c)+1e-3),-1))
            neg_loss=torch.mean(torch.sum(n_data*(recon_batch.detach()*n_data-0)**2/(torch.exp(c)+1e-3),-1))
            loss_r = (args.alpha * pos_loss + neg_loss) + args.beta*torch.mean(torch.sum(((args.alpha*data + n_data) * c), -1)) + args.gamma * torch.mean(torch.sum(((args.alpha*data + n_data) * (c ** 2)), -1))
        loss_r.backward()
        optimizer.step()
        loss_r_list.append(loss_r.item())

        c_list.append((torch.mean(torch.abs(c))).item())
    etime = time.time()
    print('one_epoch done! cost time:{:4.2f} vae loss:{:4.2f}'.format(etime-stime,float(np.mean(loss_r_list))))
    print('c',np.mean(c_list))



print("AUR Experiments")
print(args)
best_p=0
counter=0
if args.stage=='backbone':
    early_stop=10
else:
    early_stop=10
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    #
    train_one_epoch_vae()
    if counter>early_stop:
        break
    if epoch%2==1:
        counter+=1
        print('-' * 89)
        print('epoch:',epoch)
        sl_r20, ul_r20, sl_n20, ul_n20 = evaluate_fast_all(model, 20,uratio=uratio)
        print("Baseline Overall Evaluation:R@20 {:.4f},N@20 {:.4f}".format(sl_r20.item(),sl_n20.item()))
        print("AUR Overall Evaluation:R@20 {:.4f},N@20 {:.4f}".format(ul_r20.item(), ul_n20.item()))
        # f = args.save
        if args.stage=='backbone':
            metric=sl_r20
            f=args.data + '_' + args.model + '.pkl'
        else:
            metric=ul_r20
            f=args.data+'_'+args.model+'_uncertain.pkl'
        if metric>best_p:
            torch.save(model.state_dict(), f)
            print('save succeed!')
            best_p=metric
            counter=0
        sl_r20, ul_r20, sl_n20, ul_n20 = evaluate_fast_t1(model, 20,sep_,uratio=uratio)
        print("Baseline Tail Absolute Evaluation:R@20 {:.4f},N@20 {:.4f}".format(sl_r20.item(),sl_n20.item()))
        print("AUR Tail Absolute Evaluation:R@20 {:.4f},N@20 {:.4f}".format(ul_r20.item(), ul_n20.item()))
        sl_r20, ul_r20, sl_n20, ul_n20 = evaluate_fast_t2(model, 20, sep_,uratio=uratio)
        print("Baseline Tail Relative Evaluation:R@20 {:.4f},N@20 {:.4f}".format(sl_r20.item(),sl_n20.item()))
        print("AUR Tail Relative Evaluation:R@20 {:.4f},N@20 {:.4f}".format(ul_r20.item(), ul_n20.item()))

save_model=torch.load(f,map_location='cpu')
model_dict =  model.state_dict()

state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}

print(state_dict.keys())

model_dict.update(state_dict)
model.load_state_dict(model_dict)

sl_r50, ul_r50, sl_n50, ul_n50 = evaluate_fast_all(model, 50)
print("Baseline Overall Evaluation:R@50 {:.4f},N@50 {:.4f}".format(sl_r50.item(), sl_n50.item()))
print("AUR Overall Evaluation:R@50 {:.4f},N@50 {:.4f}".format(ul_r50.item(), ul_n50.item()))
sl_r50, ul_r50, sl_n50, ul_n50 = evaluate_fast_t1(model, 50, sep_)
print("Baseline Tail Absolute Evaluation:R@50 {:.4f},N@50 {:.4f}".format(sl_r50.item(), sl_n50.item()))
print("AUR Tail Absolute Evaluation:R@50 {:.4f},N@50 {:.4f}".format(ul_r50.item(), ul_n50.item()))
sl_r50, ul_r50, sl_n50, ul_n50 = evaluate_fast_t2(model, 50, sep_)
print("Baseline Tail Relative Evaluation:R@50 {:.4f},N@50 {:.4f}".format(sl_r50.item(), sl_n50.item()))
print("AUR Tail Relative Evaluation:R@50 {:.4f},N@50 {:.4f}".format(ul_r50.item(), ul_n50.item()))