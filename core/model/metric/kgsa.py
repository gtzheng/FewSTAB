import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNet(nn.Module):
    def __init__(self, dims):
        super(FeedForwardNet, self).__init__()
        self.ndims = len(dims)
        self.act = F.relu
        for i in range(1, self.ndims):
            setattr(self, 'linear%d' % (i), nn.Linear(dims[i-1],dims[i]))
    
    def forward(self, x, weights=None):
        out = x
        if weights is None:
            for i in range(1, self.ndims-1):
                out = getattr(self, 'linear%d' % (i))(out)
                out = self.act(out)
            out = getattr(self, 'linear%d' % (self.ndims-1))(out)
            
        else:
            for i in range(1, self.ndims-1):
                out = F.linear(out, weights['linear%d.weight' % i], weights['linear%d.bias' % i])
                out = self.act(out)
            out = F.linear(out, weights['linear%d.weight' % (self.ndims-1)], weights['linear%d.bias' % (self.ndims-1)])

        return out


class KGSAModule(nn.Module):
    def __init__(self, n_vecs, fea_dim, dims, dropout=0.2, r=1.0):
        super(KGSAModule, self).__init__()
        self.kernel = FeedForwardNet(dims)
        n_base = dims[0]
        assert n_vecs == n_base, 'incompatible dimensions'
        self.n_vecs = n_vecs
        self.fea_vecs = nn.Parameter(torch.normal(0,0.01, size=(n_vecs, fea_dim)))
        self.dropout = nn.Dropout(dropout)
        self.r = r

    def proj_all(self, v1):
        nkb = self.fea_vecs
        proj_v1 = torch.matmul(torch.matmul(v1,nkb.T),torch.inverse(torch.matmul(nkb,nkb.T)+self.r))
        return proj_v1
        
    def forward(self, x): # x.shape = (n_batch, n_num, n_fea)
        n_batch, n_num = x.shape[0:2]
        projection = self.proj_all(x)  # n_batch, n_num, n_constr
        n_batch, n_num, n_constr = projection.shape
        steps = torch.relu(self.kernel(projection.view(-1,n_constr)))# n_batch * n_way, n_constr
        steps = steps.reshape(n_batch, n_num, n_constr)
        steps = self.dropout(steps)
        direction = torch.matmul(steps, F.normalize(self.fea_vecs,dim=-1))
        new_x = x - direction # addition
        return new_x

class KGSA(nn.Module):
    def __init__(self, encoder, encoder_args={},n_vecs=5, n_hidden=20, n_layer=2,dropout=0.2,
                 temp=1., temp_learnable=False):
        super(KGSA, self).__init__()
        self.encoder = 
        if type(self.encoder.out_dim) == tuple:
            self.fea_dim = self.encoder.out_dim[0]
        else:
            self.fea_dim = self.encoder.out_dim

        dims = [n_vecs] + [n_hidden] * n_layer + [n_vecs]
        self.kgsa = KGSAModule(n_vecs, self.fea_dim, dims,dropout)
    
        self.temp_learnable = temp_learnable
        if temp_learnable == False:
            self.temp = torch.tensor(temp).cuda()
        else:
            self.temp = nn.Parameter(torch.tensor(temp))
    

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        if len(shot_shape) == 4:
            n_batch, n_way, n_shot, n_aug = shot_shape
        else:
            n_batch, n_way, n_shot = shot_shape

        
        n_query = query_shape[1] // n_way
        x_shot = x_shot.reshape(-1, *img_shape).contiguous()
        x_query = x_query.reshape(-1, *img_shape).contiguous()

        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):] # num, n_fea
        n_fea = x_shot.shape[-1]
        fx_query_batch = x_query.reshape(*query_shape,-1) # n_batch, n_query, n_aug, n_fea
        
        fx_support_batch = x_shot.reshape(*shot_shape,-1) #n_batch, n_way, n_shot, n_aug, n_fea
    
            
        loss_all = 0.0
        logits_all = []
        for nb in range(n_batch):
            fx_support = fx_support_batch[nb:nb+1] # 1, n_way, n_shot, n_aug, n_fea
            if len(shot_shape) == 4:
                fx_support = fx_support.mean(3)
            fx_query = fx_query_batch[nb:nb+1] # # 1, n_query, n_aug, n_fea
    
            fx_support = self.kgsa(fx_support.reshape(1,-1,n_fea))
            prototypes = fx_support.reshape(1, n_way, -1, n_fea).mean(2)
       
            if len(shot_shape) == 4:
                fx_query = fx_query.mean(2)
            fx_query = self.kgsa(fx_query.reshape(1,-1,n_fea)) # 1, n_query*n_aug, n_fea
            
            logits = torch.bmm(F.normalize(fx_query,dim=-1), F.normalize(prototypes,dim=-1).permute(0,2,1)) * self.temp # cosine distance

            logits = logits.reshape(-1, n_way)
            logits_all.append(logits)
            label = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1).cuda()
            loss = F.cross_entropy(logits, label)
            loss_all = loss_all + loss

        return torch.cat(logits_all,dim=0), loss_all

    def evaluate(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        if len(shot_shape) == 4:
            n_batch, n_way, n_shot, n_aug = shot_shape
        else:
            n_batch, n_way, n_shot = shot_shape
            n_aug = 1

        n_query = query_shape[1] // n_way
        x_shot = x_shot.reshape(-1, *img_shape).contiguous()
        x_query = x_query.reshape(-1, *img_shape).contiguous()
        if n_aug <= 5:
            x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        else:
            batch_sample = torch.cat([x_shot, x_query], dim=0)
            batch_size = 512
            num_sample = len(batch_sample)
            n_iters = num_sample // batch_size
            remain_num = num_sample - n_iters * batch_size
            x_tot = []
            for idx in range(n_iters):
                x_tot.append(self.encoder(batch_sample[idx*batch_size:(idx+1)*batch_size]))
            if remain_num > 0:
                x_tot.append(self.encoder(batch_sample[-remain_num:]))
            x_tot = torch.cat(x_tot,dim=0)
        
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):] # num, n_fea
        n_fea = x_shot.shape[-1]
        fx_query_batch = x_query.reshape(*query_shape,-1) # n_batch, n_query, n_aug, n_fea
        
        fx_support_batch = x_shot.reshape(*shot_shape,-1) #n_batch, n_way, n_shot, n_aug, n_fea
    
            
        loss_all = 0.0
        logits_all = []
        for nb in range(n_batch):
            fx_support = fx_support_batch[nb:nb+1] # 1, n_way, n_shot, n_aug, n_fea
            if len(shot_shape) == 4:
                fx_support = fx_support.mean(3)
            fx_query = fx_query_batch[nb:nb+1] # # 1, n_query, n_aug, n_fea
    
            fx_support = self.kgsa(fx_support.reshape(1,-1,n_fea))
            prototypes = fx_support.reshape(1, n_way, -1, n_fea).mean(2)
            
            if len(shot_shape) == 4:
                fx_query = fx_query.mean(2)
            fx_query = self.kgsa(fx_query.reshape(1,-1,n_fea)) # 1, n_query*n_aug, n_fea
            
            logits = torch.bmm(F.normalize(fx_query,dim=-1), F.normalize(prototypes,dim=-1).permute(0,2,1)) * self.temp # cosine distance
    
            logits = logits.reshape(-1, n_way)
            logits_all.append(logits)
            label = torch.arange(n_way).unsqueeze(1).expand(n_way, n_query).reshape(-1).cuda()
            loss = F.cross_entropy(logits, label)
            loss_all = loss_all + loss
        return torch.cat(logits_all,dim=0), loss_all 