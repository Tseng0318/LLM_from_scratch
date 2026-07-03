import torch
import torch.nn as nn
from einops import einsum, rearrange

def silu(x):
    return x * torch.sigmoid(x)



class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ## Make an empty tensor for W. (y = x W^T)
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)

        std = (2.0/(out_features+in_features))**0.5

        # Fill W in-place with truncated normal, clipped at ±3*std
        nn.init.trunc_normal_(W, mean=0.0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(W)

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)   # [num_emdedding, emdedding_dim]
        nn.init.trunc_normal_(W, mean=0.0, std=1, a=-3, b=3)  # σ=1, clip ±3
        self.weight = nn.Parameter(W)

    def forward(self, token_ids):
        return self.weight[token_ids]      # index into the table
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):# Example: a = [3, 4], d=2.
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x):
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)+self.eps) # # squares: [9, 16]
        result = x/rms*self.weight # ← x / rms * self.weight
        return result.to(in_type)
# mean: (9+16)/2 = 12.5
# + eps (tiny, ignore for now): 12.5
# sqrt: ≈ 3.54
# So RMS(a) ≈ 3.54.

# Step 2 — divide by that size. Each element gets divided by RMS(a):

# a / RMS = [3/3.54, 4/3.54] = [0.85, 1.13]
# Now the vector has RMS ≈ 1 — it's been scaled to a standard magnitude, regardless of how big or small it started.

# Step 3 — apply learnable gain. Multiply elementwise by g (the self.weight, one value per channel, length d):

# if g = [1, 1] (the init), output stays [0.85, 1.13]

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype) 
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) 

    def forward(self, x):
        return self.w2(silu(self.w1(x))*self.w3(x))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()
        # angle(i,k) = i/theta**(2k/d_k)
        exponents = torch.arange(0, d_k, 2, device=device) / d_k
        inv_freq = theta**(-exponents)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) # i
        angles = einsum(positions, inv_freq, "i, k -> i k")
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x:torch.tensor, token_positions:torch.tensor) -> torch.tensor:
        cos = self.cos_cache[token_positions]    # (..., seq_len, d_k/2)
        sin = self.sin_cache[token_positions]    # (..., seq_len, d_k/2)

        x_even = x[..., 0::2]    # (..., seq_len, d_k/2)  — x0, x2, x4, ...
        x_odd  = x[..., 1::2]    # (..., seq_len, d_k/2)  — x1, x3, x5, ...

        x_even_rot = x_even*cos - x_odd*sin         # x_even*cos − x_odd*sin
        x_odd_rot  = x_even*sin + x_odd*cos          # x_even*sin + x_odd*cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_even_rot
        out[..., 1::2] = x_odd_rot
        return out
    
def softmax(x, dim):
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Attention(Q,K,V)=softmax((QK^T)/d_k)V
    # Q=nd K=md V=md
    d_k = Q.shape[-1]
    scores = einsum(Q,K, "... n d, ... m d ->... n m")/(d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    weights = softmax(scores, dim=-1) # row
    return einsum(weights, V, "... n m, ... m d -> ... n d")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype) # needed to concat q k v
        self.rope = rope
    
    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2] # (..., seq_len, d_model)

        #1 project
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Split heads: (..., seq, h*d_k) -> (..., h, seq, d_k) 
        # eg Q = [[1,0,0,1],
        #         [0,1,1,0]] seq*d_model -> head, seq, d_k
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # 3. apply RoPE
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device) # assume the tokens are in their natural position
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # 4. Mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)) # lowwer triangle
        #          key0  key1  key2  key3
        # query0:   T     F     F     F      ← token 0 sees only itself
        # query1:   T     T     F     F      ← token 1 sees tokens 0,1
        # query2:   T     T     T     F      ← token 2 sees tokens 0,1,2
        # query3:   T     T     T     T      ← token 3 sees everything before + itself
        # 5. Atttention
        out = scaled_dot_product_attention(Q, K, V, mask)

        # 6. rearrange
        out = rearrange(out, "... h seq d -> ... seq (h d)") 

        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self,d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers,
                 num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # one shared RoPE for all layers
        d_k = d_model // num_heads
        rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_k, max_seq_len=context_length, device=device)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)        # (batch, seq, d_model)
        for layer in self.layers:
            x = layer(x)                            # each block, in order
        x = self.ln_final(x)
        return self.lm_head(x)                      # (batch, seq, vocab_size)


