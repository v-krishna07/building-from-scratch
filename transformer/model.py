import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model=d_model
        self.x = nn.Embedding(vocab_size,d_model).requires_grad_(True)
    def forward(self,z):
        out = self.x(z)/self.d_model**0.5
        return out

class PositionEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # position
        position = torch.arange(1,max_len+1, dtype=torch.float16).unsqueeze(1)
        mat = torch.zeros(max_len, d_model).to(device)
        x = torch.sin(position/((10000**torch.tensor(torch.arange(0,d_model,2)/d_model))))
        y = torch.cos(position/((10000**torch.tensor(torch.arange(1,d_model,2)/d_model))))
        mat[:,::2] = x
        mat[:,1::2] = y
        self.pos = mat.unsqueeze(0)
        self.register_buffer('pe',pos)

    def forward(self,x):
        # self.pos => shape=> (1,max_len_d_model)
        return x + self.pos[:,:x.shape[1],:]

# class LayerNorm(nn.Module):
#     def __init__(self, layer,eps):
#         super().__init__()
#         self.eps=eps
#         self.layer = layer
#         m = torch.mean(layer, dim=-1, keepdim=True)
#         var = torch.var(layer, dim=-1, keepdim=True)
#         gamma = torch.ones(layer.shape[-2]).unsqueeze(1)
#         beta = torch.zeros(layer.shape[-2]).unsqueeze(1)
#         self.layerNorm = gamma*((layer-m)/((var+eps)**0.5))+ beta

#     def forward(self):
#         return self.layerNorm

class LayerNorm(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self):
        return nn.LayerNorm(self.layer, eps = 1e-5, device=device, dtype=torch.float16)


class feedforward(nn.Module):
    def __init__(self, in_features,dff, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        l = nn.Linear(in_features=in_features, out_features=dff)
        l2 = nn.Linear(in_features=dff, out_features=in_features)
        # relu = nn.ReLU()
    
    def forward(self,x):
        return self.l2(self.dropout(self.relu(self.l(x))))

class MHA(nn.Module):
    def __init__(self,batch,seq_len,d_model, h):
        super().__init__()
        self.h = h
        self.d_model = d_model
        self.w_q = torch.randn(d_model,d_model)
        self.w_k = torch.randn(d_model,d_model)
        self.w_v = torch.randn(d_model,d_model)

        self.w_o = torch.randn(d_model, d_model)

    def forward(self,q,k,v,mask=None):
        h  =self.h
        q_ = q@w_q
        k_ = k@w_q
        v_ = v@w_q # seq_len , d_model

        assert d_model % h==0

        q = q_.view(q_[0],-1,h, d_model//h).transpose(-2,-3) # batch, head, seq_len, dk
        k = k_.view(k_[0],-1,h, d_model//h).transpose(-2,-3)
        v = v_.view(v_[0],-1,h, d_model//h).transpose(-2,-3)

        x = nn.Softmax(dim=-1)
        z = (x(torch.tensor((q@(k.transpose(-2,-1)))/((d_model//h)**0.5)))@v).transpose(-2,-3)
        z = z.contiguous().view(z.shape[0],-1,d_model)
        z = z@self.w_o
        return z

class Residual(nn.Module):
    def __init__(self,dropout):
        self.dropout = nn.Dropout(dropout)
    def forward(self,x, fn=None):
        s = self.dropout(x)
        if fn is not None:
            return fn(x)
        return x
