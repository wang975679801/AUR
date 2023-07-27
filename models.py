import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy import sparse
class MultiVAE_Linear(nn.Module):
    #Multi-VAE model for AUR

    def __init__(self, p_dims, q_dims=None, dropout=0.5,activate_function=None,user_uncertain=False,n_user=100,stage='backbone'):
        super(MultiVAE_Linear, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]
        self.user_uncertain=user_uncertain
        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.user_uncertain_embedding = nn.Embedding(n_user, p_dims[1])
        self.drop = nn.Dropout(dropout)
        self.activate_function=activate_function
        self.c_layer=nn.ModuleList([
            nn.Linear(self.q_dims[0] , 1024,bias=False),
            nn.Linear(1024, self.q_dims[0],bias=False)
        ])
        self.init_weights()
        print(self.activate_function)
        for layer in self.c_layer:
            nn.init.normal_(layer.weight,std=1e-2,mean=0)
        self.stage=stage
    def init_c(self):
        for layer in self.c_layer:
            nn.init.normal_(layer.weight,std=1e-2,mean=0)
    def get_confidence(self,input,input_idx=None):
        c = F.normalize(input)
        # print(input_idx.max())
        ###Use linear projection to represent the aggregation
        for i, layer in enumerate(self.c_layer):
            c=layer(c)
            if i != len(self.c_layer) - 1:
                if self.user_uncertain and input_idx:
                    c = c + self.user_uncertain_embedding.weight[input_idx]
                if self.activate_function=='relu':
                    c=torch.relu(c)
                elif self.activate_function=='tanh':
                    c=torch.tanh(c)
                elif self.activate_function=='elu':
                    m=torch.nn.ELU()
                    c=m(c)
                    # print(1)
                elif self.activate_function=='linear':
                    c=c
                else:
                    raise
        return c
    def forward(self, input,input_idx=None):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        res=self.decode(z)
        if self.stage=='backbone':
            c=torch.zeros_like(res,device=res.device)
        elif self.stage=='uncertain':
            c=self.get_confidence(input,input_idx)
        else:
            raise
        return res,c



    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar



    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
class LGCN_Linear(nn.Module):
    """
    MF Model
    """

    def __init__(self,user_item_net, dataset=None, n_layers=3, hidden_dim=128,num_user=100,num_item=100 , dim=1024,user_uncertain=False,stage='backbone'):
        super(LGCN_Linear, self).__init__()
        self.num_item=num_item
        self.num_user=num_user
        self.hidden_dim=hidden_dim
        self.user_embedding=nn.Embedding(num_user,hidden_dim)
        self.item_embedding=nn.Embedding(num_item,hidden_dim)
        self.user_item_net=user_item_net
        self.n_layers=n_layers
        self.stage=stage
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        self.user_uncertain_embedding=nn.Embedding(num_user,dim)
        nn.init.normal(self.user_uncertain_embedding.weight,std=0.01)
        self.c_layer=nn.ModuleList([
            nn.Linear(self.num_item , dim,bias=False),
            # nn.Linear(dim,dim),
            nn.Linear(dim,self.num_item,bias=False)
        ])
        self.dataset=dataset
        self.user_uncertain=user_uncertain
        self.adj_tensor=None
        # for layer in self.c_layer:
        #     nn.init.normal_(layer.weight,std=1e-2,mean=0)
        # self.c_layers=nn.ModuleList([
        #     nn.Linear(self.num_item , 1024),
        #     nn.Linear(1024, 1024),
        #     nn.Linear(1024,1024),
        #     nn.Linear(1024, self.num_item)
        # ])
        # self.norm_layers=nn.ModuleList([
        #     nn.BatchNorm1d(1024),
        #     nn.BatchNorm1d(1024),
        #     nn.BatchNorm1d(1024),
        # ])
    def init_c(self):
        for layer in self.c_layer:
            nn.init.normal_(layer.weight,std=1e-2,mean=0)
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    def init_weight(self):
        for layer in self.c_layer:
            nn.init.normal_(layer.weight,std=1e-2,mean=0)
    def get_adj(self):
        try:
            norm_adj=sparse.load_npz(self.dataset+'_pre_adj.npz')
            print('pre adj load success')
        except:
            n_user=self.num_user
            n_item=self.num_item
            adj_mat = sparse.dok_matrix((n_user + n_item, n_user + n_item))
            adj_mat = adj_mat.tolil()
            adj_mat[n_user:, :n_user] = self.user_item_net.T
            adj_mat[:n_user, n_user:] = self.user_item_net
            adj_mat = adj_mat.todok()
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sparse.diags(d_inv)
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            print('adj done!')
            sparse.save_npz(self.dataset+'_pre_adj.npz',norm_adj)
            # adj_tensor = torch.FloatTensor(norm_adj.toarray())
        self.adj_tensor = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.adj_tensor=self.adj_tensor.coalesce().cuda()
        print('adj_matrix has been computed')
    def get_confidence(self,input,input_idx=None):
        c = F.normalize(input)
        # c = self.drop(c)
        for i, layer in enumerate(self.c_layer):
            c=layer(c)
            if i != len(self.c_layer) - 1:
                if input_idx:
                    c=c+self.user_uncertain_embedding.weight[input_idx]
                # c =F.relu(c)
                pass
                # c=self.norm_layers[i](c)
                # c=self.drop(c)
        return c
    def forward(self, input,input_idx):
        u_emb=self.user_embedding
        i_emb=self.item_embedding
        # print(u_emb.weight.shape,i_emb.weight.shape)
        all_emb=torch.cat([u_emb.weight,i_emb.weight],0)
        embs = all_emb
        if self.adj_tensor!=None:
            for i in range(self.n_layers):
                all_emb = torch.sparse.mm(self.adj_tensor, all_emb)
                embs=embs+all_emb
        # embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = embs/(self.n_layers+1)
        users, items = torch.split(light_out, [self.num_user, self.num_item])
        user_batch=users[input_idx]
        output=torch.mm(user_batch,items.T)
        if self.stage=='backbone':
            c=torch.zeros_like(output,device=output.device)
        elif self.stage=='uncertain':
            if self.user_uncertain:
                c = self.get_confidence(input, input_idx)
            else:
                c = self.get_confidence(input)
        else:
            raise
        # if self.user_uncertain:
        #     c=self.get_confidence(input,input_idx)
        # else:
        #     c=self.get_confidence(input)
        return output,c

