import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange 
from torch.distributions.normal import Normal
import numpy as np
from collections import OrderedDict
import math

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_feat = out_feat or in_feat
        h_feat = h_feat or in_feat
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, mlp_ratio, num_experts, noisy_gating=True, use_experts=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = use_experts
        # instantiate experts
        self.experts = nn.ModuleList([Mlp(input_size, h_feat=int(input_size*mlp_ratio), out_feat=output_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.randn(2*input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(2*input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts) 

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, prompt):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """ 
        
        # import pdb 
        # pdb.set_trace() 
        
        B, C, H, W = x.shape
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand_as(x) 
        
        
        x = rearrange(x, 'b c h w -> (b h w) c') 
        prompt = rearrange(prompt, 'b c h w -> (b h w) c') 
        
        x_gating = torch.cat((x, prompt), dim=1) #[B, 2C, H, W]
        
        gates, load = self.noisy_top_k_gating(x_gating, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs) 
        
        y = rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)
        
        return y, loss


class RIN(nn.Module):
    def __init__(self, in_dim, atom_num=16, atom_dim=256):
        super(RIN, self).__init__()
        
        # Condtion network 
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3,3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1)) 
        
        self.lastOut = nn.Linear(32, atom_num) 
        self.act = nn.GELU()
        
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out) 
        logits = F.softmax(out, -1) 
        out = logits @ self.dictionary 
        out = self.act(out)
        return out 



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Channel_Routing(nn.Module):

    def __init__(self, atom_dim, dim):
        super(Channel_Routing, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True) 

    def forward(self, x, prompt):
        gating_factors = torch.sigmoid(self.fc(prompt))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)

        out = x * self.gamma + self.beta  
        out = out * gating_factors 
             
        return x + out


class Spatial_Routing(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor):
        super(Spatial_Routing, self).__init__() 
        
        self.fc = nn.Linear(atom_dim, dim) 
        self.moe = MoE(dim, dim, mlp_ratio=ffn_expansion_factor, num_experts=4, noisy_gating=True, use_experts=2) 

    def forward(self, x, prompt): 
        d = self.fc(prompt) 
        out, loss = self.moe(x, d) 
        return out + x, loss




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

## ------ Spatial Attention --------------
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
##---------- Dual Attention Unit ----------
def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class DualAttention(nn.Module):
    def __init__(
            self,infeat, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualAttention, self).__init__()
        modules_body = [conv(infeat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

#############################################################################################################
class CAFF(nn.Module):
    def __init__(self, dim, d):
        super(CAFF, self).__init__()

        # self.num_features = num_features
        self.mlp_vis = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, 512, 1, bias=False)
        )
        # self.fc_gamma = nn.Sequential(OrderedDict([
        #     ('linear1',nn.Linear(cond_dim, cond_dim)),
        #     ('relu1',nn.ReLU(inplace=True)),
        #     ('linear2',nn.Linear(cond_dim, num_features)),
        #     ]))
        # self.fc_beta = nn.Sequential(OrderedDict([
        #     ('linear1',nn.Linear(cond_dim, cond_dim)),
        #     ('relu1',nn.ReLU(inplace=True)),
        #     ('linear2',nn.Linear(cond_dim, num_features)),
        #     ]))

    #     self._initialize()

    # def _initialize(self):
    #     nn.init.zeros_(self.fc_gamma.linear2.weight.data)
    #     nn.init.ones_(self.fc_gamma.linear2.bias.data)
    #     nn.init.zeros_(self.fc_beta.linear2.weight.data)
    #     nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, c):
        x_vis = self.mlp_vis(x)
        text_features = c.view(1,c.shape[1],1,1).expand_as(x_vis)
        # gamma = self.fc_gamma(c).view(-1, self.num_features, 1, 1)
        # beta = self.fc_beta(c).view(-1, self.num_features, 1, 1)
        return text_features * x_vis + x_vis

class DEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.CAFF = CAFF(in_ch, in_ch*8)
        # self.DCBlk = DCBlock(out_ch)
        self.conv2 = nn.Conv2d(512, out_ch, 3, 1, 1)

        # self.fc = nn.Sequential(OrderedDict([
        #     ('linear2',nn.Linear(256*4, out_ch)),
        #     ('sigmoid',nn.Sigmoid()),
        #     ]))

    #     self._initialize()

    # def _initialize(self):
    #     nn.init.zeros_(self.fc.linear2.weight.data)
    #     nn.init.zeros_(self.fc.linear2.bias.data)

    def forward(self, h, y):
        # h = F.interpolate(h, scale_factor=2)
        h = self.conv(h)
        h_m = nn.LeakyReLU(0.2, inplace=True)(self.CAFF(h, y))
        h_m= self.conv2(h_m)
        # h_s = nn.LeakyReLU(0.2, inplace=True)(self.DCBlk(h, mask, y))
        # weights = self.fc(ti).unsqueeze(-1).unsqueeze(-1)
        # h = weights*h_m + (1-weights)*h_s
        h = h_m + h
        return h

class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

##---------- AMIR -----------------------
class NetG(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 24,
        num_blocks = [2,2,2,2], 
        # num_blocks = [1,1,1,1], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(NetG, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) 
        self.patch_embed_b = OverlapPatchEmbed(inp_channels, dim) 
        
        atom_dim = 24
        atom_num = 32 
        # self.dict_generator = RIN(in_dim=inp_channels, atom_num=atom_num, atom_dim=atom_dim) 
        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.fuse1 = DEBlock(dim, dim)
        self.fuse1_b = DEBlock(dim, dim)
        self.fuse2 = DEBlock(dim*2, dim*2)
        self.fuse2_b = DEBlock(dim*2, dim*2)
        self.fuse3 = DEBlock(dim*4, dim*4)
        self.fuse3_b = DEBlock(dim*4, dim*4)
        
        self.spatial_routing_encoder_level1 = DualAttention( atom_dim, dim) 
        self.spatial_routing_encoder_level2 = DualAttention( atom_dim*2, int(dim*2**1))
        self.spatial_routing_encoder_level3 = DualAttention( atom_dim*4, int(dim*2**2))
        self.spatial_routing_encoder_level1_b = DualAttention( atom_dim, dim) 
        self.spatial_routing_encoder_level2_b = DualAttention( atom_dim*2, int(dim*2**1))
        self.spatial_routing_encoder_level3_b = DualAttention( atom_dim*4, int(dim*2**2))


        
        self.channel_routing_latent = DualAttention( atom_dim*8, int(dim*2**3))
        self.channel_routing_latent_b = DualAttention( atom_dim*8, int(dim*2**3))

        self.channel_routing_decoder_level3 = DualAttention( atom_dim*4, int(dim*2**2)) 
        self.channel_routing_decoder_level2 = DualAttention( atom_dim*2, int(dim*2**1)) 
        self.channel_routing_decoder_level1 = DualAttention( atom_dim*2, int(dim*2**1)) 
        self.channel_routing_decoder_level3_b = DualAttention( atom_dim*4, int(dim*2**2)) 
        self.channel_routing_decoder_level2_b = DualAttention( atom_dim*2, int(dim*2**1)) 
        self.channel_routing_decoder_level1_b = DualAttention( atom_dim*2, int(dim*2**1)) 

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_level1_b = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        # B
        self.down1_2_b = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2_b = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3_b = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3_b = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4_b = Downsample(int(dim*2**2)) ## From Level 3 to Level 4

        self.latent_b = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])



        self.up4_3 = Upsample(int(dim*2**4)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3 + 96), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output3 = nn.Conv2d(int(dim*4**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.feature_fusion_1 = Fusion_Embed(embed_dim=dim)
        self.feature_fusion_2 = Fusion_Embed(embed_dim=dim*2)
        self.feature_fusion_3 = Fusion_Embed(embed_dim=dim*2*2)

    def forward(self, vis, ir, text_features): 
        
        # VIS encoder
        inp_enc_level1 = self.patch_embed(vis) #[1, 42, 512, 640]
        inp_enc_level1 = self.spatial_routing_encoder_level1(inp_enc_level1) 
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 
        text_image1 = self.fuse1(out_enc_level1, text_features)


        inp_enc_level2 = self.down1_2(text_image1) 
        inp_enc_level2 = self.spatial_routing_encoder_level2(inp_enc_level2) 
        out_enc_level2 = self.encoder_level2(inp_enc_level2) 
        text_image2 = self.fuse2(out_enc_level2, text_features)


        inp_enc_level3 = self.down2_3(text_image2) 
        inp_enc_level3 = self.spatial_routing_encoder_level3(inp_enc_level3) 
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        text_image3 = self.fuse3(out_enc_level3, text_features)


        inp_enc_level4 = self.down3_4(text_image3)        
        inp_enc_level4 = self.channel_routing_latent(inp_enc_level4)
        latent = self.latent(inp_enc_level4) 

        # IR encoder
        inp_enc_level1_b = self.patch_embed_b(ir) #[1, 42, 512, 640]
        inp_enc_level1_b = self.spatial_routing_encoder_level1_b(inp_enc_level1_b) 
        out_enc_level1_b = self.encoder_level1_b(inp_enc_level1_b) 
        text_image1_b = self.fuse1_b(out_enc_level1_b, text_features)


        inp_enc_level2_b = self.down1_2(text_image1_b) 
        inp_enc_level2_b = self.spatial_routing_encoder_level2_b(inp_enc_level2_b) 
        out_enc_level2_b = self.encoder_level2_b(inp_enc_level2_b) 
        text_image2_b = self.fuse2(out_enc_level2_b, text_features)


        inp_enc_level3_b = self.down2_3_b(text_image2_b) 
        inp_enc_level3_b = self.spatial_routing_encoder_level3_b(inp_enc_level3_b) 
        out_enc_level3_b = self.encoder_level3_b(inp_enc_level3_b) 
        text_image3_b = self.fuse3(out_enc_level3_b, text_features)


        inp_enc_level4_b = self.down3_4_b(text_image3_b)        
        inp_enc_level4_b = self.channel_routing_latent_b(inp_enc_level4_b)
        latent_b = self.latent_b(inp_enc_level4_b)
        
        text1 = self.feature_fusion_1(text_image1, text_image1_b)
        text2 = self.feature_fusion_2(text_image2, text_image2_b)
        text3 = self.feature_fusion_3(text_image3, text_image3_b)
        latent, latent_b = self.cross_attention(latent, latent_b)
        # Decoder          
        inp_dec_level3 = self.up4_3(torch.concat([latent, latent_b], dim=1))
        inp_dec_level3 = torch.cat([inp_dec_level3, text3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        inp_dec_level3 = self.channel_routing_decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) # ([1, 96, 128, 160])
        out_3 = self.output3(out_dec_level3)
        

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, text2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        inp_dec_level2 = self.channel_routing_decoder_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) # ([1, 48, 256, 320])
        out_2 = self.output2(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, text1], 1) 
        inp_dec_level1 = self.channel_routing_decoder_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1) 
        
        
        out_dec_level1 = self.refinement(out_dec_level1) 

        out_dec_level1 = self.output(out_dec_level1) # [1, 1, 512, 640]

        # if self.training:
        #     return out_dec_level1, loss_importance 
        # else: 
        return out_dec_level1 , out_2, out_3 # [1, 1, 512, 640], [1, 1, 256, 320], [1, 1, 128, 160]



class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

class D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bn=True):
        super(D_Block, self).__init__()
        self.bn = bn
        self.conv = nn.Conv2d(in_ch, 512, kernel_size, stride, padding, bias=False)
        self.out = nn.Conv2d(512, out_ch, kernel_size, stride, padding, bias=False)
        if bn==True:
            self.batchnorm = nn.BatchNorm2d(512)
        else:
            self.batchnorm = None

    def forward(self, x, c):
        h = self.conv(x)
        if self.bn==True:
            h = self.batchnorm(h)
        h = nn.LeakyReLU(0.2, inplace=True)(h)
        text_features = c.view(1,c.shape[1],1,1).expand_as(h)
        h_t = h * text_features
        h_t = self.out(h_t)
        h = h_t + x
        return h
    
class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()

        self.block0 = D_Block(1, 48, 3, 1, 1, bn=False)#128
        self.block1 = D_Block(49, 49, 3, 1, 1)#64
        self.block2 = D_Block(50, 50, 3, 1, 1)#32
        # self.block3 = D_Block(48, 48, 4, 2, 1)#16
        # self.block4 = D_Block(48, 48, 4, 2, 1)#8
        # self.block5 = D_Block(48, 48, 4, 2, 1)#4
        self.conv9 = nn.Conv2d(50, 1, 3, 1, 1)

    def forward(self,out1, out2, out3, features):

        h = self.block0(out1, features)
        h = F.interpolate(h, scale_factor=0.5, mode='bilinear');
        h = self.block1(torch.cat((out2, h), 1), features)
        h = F.interpolate(h, scale_factor=0.5, mode='bilinear');
        h = self.block2(torch.cat((out3, h), 1), features)
        # h = self.block3(h)
        # h = self.block4(h)
        # h = self.block5(h)
        out = self.conv9(h)
        return out


def Net_G():
    return NetG()

def Net_D():
    return NetD()