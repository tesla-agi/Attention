import torch
import math
import torch.nn.functional as F
import torch.nn as nn

def scaled_dot_product_attention(query,key,value,mask=None):
    d_k=query.shape[-1] #batch,heads,seq_len,d_k
    scores=torch.matmul(query,key.transpose(-2,-1)/math.sqrt(d_k))
    if mask is not None:
        scores=scores.masked_fill(mask==0,float('-inf'))
    weights=F.softmax(scores,dim=-1)
    return torch.matmul(weights,value),weights

'''
b,h,s,d_k=2,4,10,64
Q=torch.rand(b,h,s,d_k)
K=torch.rand(b,h,s,d_k)
V=torch.rand(b,h,s,d_k)

out,weights=scaled_dot_product_attention(Q,K,V)
print(out.shape)
print(weights.shape)

'''


class MHSA(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MHSA,self).__init__()

        assert(d_model%num_heads)==0
        self.num_heads=num_heads
        self.d_model=d_model
        self.d_k=d_model//num_heads

        self.Wq=nn.Linear(d_model,d_model)
        self.Wk=nn.Linear(d_model,d_model)
        self.Wv=nn.Linear(d_model,d_model)
        self.WO=nn.Linear(d_model,d_model)

    def forward(self,x,mask=None):
        batch,seq_len,d_model=x.shape
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)

        Q=Q.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        K=K.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V=V.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)

        out,weights=scaled_dot_product_attention(Q,K,V,mask)
        out=out.transpose(1,2).contiguous().view(batch,seq_len,self.d_model)

        return out
'''
mha = MHSA(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)
out = mha(x)
print(out.shape)  # should be (2, 10, 512)
'''

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(FeedForward,self).__init__()

        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)

    def forward(self,x,mask=None):
        return self.fc2(F.relu(self.fc1(x)))



class TransformerB(nn.Module):
    def __init__(self,d_model,d_ff,num_heads,dropout=0.1):
        super().__init__()
        self.attention=MHSA(d_model,num_heads)
        self.ff=FeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        attn_out=self.attention(x,mask)
        x=self.norm1(x+self.dropout(attn_out))

        ff_out=self.ff(x)
        x=self.norm2(x+self.dropout(ff_out))

        return x


class TransformerE(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,num_layers,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList([
            TransformerB(d_model,d_ff,num_heads,dropout)
            for _ in range(num_layers)
        ])
        self.norm=nn.LayerNorm(d_model)

    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

encoder = TransformerE(d_model=512, num_heads=8, d_ff=2048, num_layers=6)
x = torch.randn(2, 10, 512)
out = encoder(x)
print(out.shape)  # should be (2, 10, 512)
