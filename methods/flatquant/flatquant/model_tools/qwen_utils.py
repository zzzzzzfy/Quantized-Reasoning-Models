import math

import torch
import torch.nn as nn

from ..quant_utils import ActivationQuantizer
from ..utils import skip_initialization
from ..function_utils import get_init_scale, get_decompose_dim
from ..trans_utils import SVDSingleTransMatrix, SVDDecomposeTransMatrix
from ..trans_utils import InvSingleTransMatrix, InvDecomposeTransMatrix
from ..trans_utils import TPTransMatrix
from ..flat_linear import FlatQuantizedLinear

from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Attention, Qwen2RotaryEmbedding,  \
                                                     apply_rotary_pos_emb, repeat_kv


class FlatQuantQwen2MLP(torch.nn.Module):
    def __init__(self, args, module: Qwen2MLP):
        super().__init__()
        self.args = args
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.act_fn = module.act_fn
        self.up_gate_quant = ActivationQuantizer(bits=args.a_bits, sym=not(args.a_asym), lac=args.lac)
        self.up_proj = FlatQuantizedLinear(args, module.up_proj, act_quantizer=self.up_gate_quant)
        self.gate_proj = FlatQuantizedLinear(args, module.gate_proj, act_quantizer=self.up_gate_quant)
        self.down_proj = FlatQuantizedLinear(args, module.down_proj, tp=True)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            self.up_smax = torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0]).to('npu') * 1e-5
            self.down_smax = torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0]).to('npu') * 1e-5
        
    def add_fq_trans(self):
        if self.args.direct_inv:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(self.up_proj.linear.weight.shape[1])
            self.up_gate_trans = DecomposeTransMatrix(up_dim_left, up_dim_right, add_diag=self.args.add_diag)
            self.down_trans_dim = self.down_proj.linear.weight.shape[1] // self.args.tp
            down_dim_left, down_dim_right = get_decompose_dim(self.down_trans_dim)
            trans_list = []
            for i in range(self.args.tp):
                trans_list.append(DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=self.args.add_diag, device="npu"))
            self.down_trans = TPTransMatrix(trans_list)
            # down_dim_left, down_dim_right = get_decompose_dim(self.down_proj.linear.weight.shape[1])
            # self.down_trans = DecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=self.args.add_diag)
        else:
            self.up_gate_trans, self.down_trans = None, None

    def _trans_forward(self, x):
        if self.up_gate_trans is not None:
            x_ts = self.up_gate_trans(x)
        else:
            x_ts = x
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)

        x_act_fn = self.act_fn(gate_states) * up_states
        if self.down_trans is not None:
            x_ts_2 = self.down_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.down_proj(x_ts_2, qa_trans=self.down_trans)
        return down_states

    def _ori_forward(self, x):
        '''origin implement: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))'''
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(self.up_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        if self.diag_init == "sq_style":
            self.down_smax = torch.maximum(self.down_smax, x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach())
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        # merge trans's diag scale
        if self.down_trans is not None and self.down_trans.get_diag_scale() is not None:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = up_weight.to(torch.float64).T.mul(self.down_trans.get_diag_scale().to(torch.float64).to(up_weight.device)).T
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            if isinstance(self.down_trans, TPTransMatrix):
                self.down_trans.disable_diag_scale()
            else:
                self.down_trans.use_diag = False

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(upw_smax, self.up_smax, alpha)
        if self.down_trans is not None:
            if isinstance(self.down_trans, TPTransMatrix):
                for i, trans in enumerate(self.down_trans.trans_list):
                    trans.diag_scale.data = get_init_scale(downw_smax[i*self.down_trans_dim: (i+1)*self.down_trans_dim], self.down_smax[i*self.down_trans_dim: (i+1)*self.down_trans_dim], alpha)
            else:
                self.down_trans.diag_scale.data = get_init_scale(downw_smax, self.down_smax, alpha)
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(self, ):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


class FlatQuantQwen2Attention(Qwen2Attention):
    def __init__(self, args, module: Qwen2Attention):
        super().__init__(module.config, module.layer_idx)
        self.args = args
        
        # 补充 rotary_emb 的定义
        self.rotary_emb = Qwen2RotaryEmbedding(config=module.config)
         
        self.qkv_quant = ActivationQuantizer(bits=args.a_bits, sym=not(args.a_asym), lac=args.lac)
        self.q_proj = FlatQuantizedLinear(args, module.q_proj, act_quantizer=self.qkv_quant)
        self.k_proj = FlatQuantizedLinear(args, module.k_proj, act_quantizer=self.qkv_quant)
        self.v_proj = FlatQuantizedLinear(args, module.v_proj, act_quantizer=self.qkv_quant)
        self.o_proj = FlatQuantizedLinear(args, module.o_proj, tp=True)
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(bits=args.q_bits, \
                                        sym=not(args.q_asym), lac=args.lac, groupsize=-1, )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(bits=args.k_bits, \
                                        sym=not(args.k_asym), lac=args.lac, groupsize=-1, )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(bits=args.v_bits, \
                                        sym=not(args.v_asym), lac=args.lac, groupsize=-1, )

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            self.ln_smax = torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]).to('npu') * 1e-5

    def add_fq_trans(self):
        if self.args.direct_inv:
            SingleTransMatrix, DecomposeTransMatrix = InvSingleTransMatrix, InvDecomposeTransMatrix
        else:
            SingleTransMatrix, DecomposeTransMatrix = SVDSingleTransMatrix, SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.ln_trans = DecomposeTransMatrix(ln_dim_left, ln_dim_right, add_diag=self.args.add_diag)
            trans_list = []
            for i in range(self.args.tp):
                trans_list.append(SingleTransMatrix(self.config.num_attention_heads // self.args.tp))
            self.o_trans = TPTransMatrix(trans_list)
            # self.o_trans = SingleTransMatrix(self.config.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kcache_trans = SingleTransMatrix(head_dim)
        else:
            self.kcache_trans = None
        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vcache_trans = SingleTransMatrix(head_dim)
        else:
            self.vcache_trans = None

    def _trans_forward_after_ln(self, hidden_states):
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states

    def _ori_forward_after_ln(self, hidden_states):
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(self.ln_smax, \
                hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].clone().detach())
        query_states = self.q_proj._ori_forward(hidden_states)
        key_states = self.k_proj._ori_forward(hidden_states)
        value_states = self.v_proj._ori_forward(hidden_states)
        return query_states, key_states, value_states

    def quant_vcache(self, value_states):
        if self.args.separate_vtrans:
            value_states = self.vcache_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states

    def quant_kcache(self, q, k):
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        # Q/K transform
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        # TODO: by default do the per-head quantizaion for k-v-cache
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value, 
            output_attentions, use_cache, cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(hidden_states)
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(hidden_states)
            
        # self.num_heads 一直在报错没有定义，翻了模型的config文件发现确实没有，加一个定义，这是transformers库版本不同造成的
        self.num_heads = self.config.num_attention_heads
        
         # self.num_key_value_heads也是同理，这是transformers库版本不同造成的
        self.num_key_value_heads = self.config.num_key_value_heads

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # ---- here do the quantization ----
        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups) # bnsh
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        # 检测不到 self.hidden_size，使用动态计算，这是transformers库版本不同造成的
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        if self._ori_mode:
            attn_output = self.o_proj._ori_forward(attn_output)
        else:
            # new foward: 
            if self.o_trans is None and self.vcache_trans is not None:
                # attn_output = self.vcache_trans(value_states)
                init_shape = attn_output.shape
                attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.config.hidden_size//self.config.num_attention_heads)
                attn_output = torch.matmul(attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output)).reshape(init_shape)
                attn_output = self.o_proj(attn_output)
            else:
                init_shape = attn_output.shape
                attn_output = attn_output.reshape(-1, self.config.num_attention_heads, self.config.hidden_size//self.config.num_attention_heads)
                attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
                    attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
                    attn_output = self.o_proj(attn_output, qa_trans=[attn_o_og_it, attn_v_og_it])
                else:
                    attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        # 出了左右和父类接收不匹配的问题，这是transformers库版本不同造成的
        # return attn_output, attn_weights, past_key_value
        return attn_output, attn_weights

    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            self.v_proj.reparameterize(qa_trans=self.ln_trans)
        else:
            self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vcache_trans.get_matrix(inv_t=True)
            self.o_proj.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = torch.cat([self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax, alpha)
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self, ):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kcache_trans is not None:
            self.kcache_trans.to_eval_mode()
        if self.vcache_trans is not None:
            self.vcache_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()


def apply_flatquant_to_qwen(args, model):
    skip_initialization()
    # Replace module with FlatQuant version
    # 如果去掉最后一层的量化，range里要-1
    for layer in range(model.config.num_hidden_layers):
        # attn
        model.model.layers[layer].self_attn = FlatQuantQwen2Attention(args, model.model.layers[layer].self_attn)
        # mlp
        model.model.layers[layer].mlp = FlatQuantQwen2MLP(args, model.model.layers[layer].mlp)
    return model
