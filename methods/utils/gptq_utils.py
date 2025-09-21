import math
import time
import tqdm
import logging

import torch
import torch.nn as nn

from . import utils
from . import quant_utils

# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        # damp = percdamp * torch.mean(torch.diag(H))
        # diag = torch.arange(self.columns, device=self.dev)
        # H[diag, diag] += damp
        # H = torch.linalg.cholesky(H)
        # H = torch.cholesky_inverse(H)
        # H = torch.linalg.cholesky(H, upper=True)
        # Hinv = H

        damp_percent = percdamp
        damp_auto_increment = 0.0015
        while 1 > damp_percent > 0:
            try:
                damp = damp_percent * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.dev)
                H[diag, diag] += damp

                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
                break
            except torch._C._LinAlgError as e:
                print(f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{ damp_auto_increment:.5f}`")
                damp_percent += damp_auto_increment

        if not (0 < damp_percent < 1):
            raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.npu.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.npu.empty_cache()
        utils.cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    '''
    logging.info('-----GPTQ Quantization-----')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
        def __getattr__(self, name):
        # 除了module和本地属性，其他都转发给self.module
            if name in ["module", "forward"]:
                return super().__getattr__(name)
            return getattr(self.module, name)
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.npu.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    quantizers = {}
    if hasattr(layers[0].self_attn.k_proj, "module"):
        sequential = [
            ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
            ['self_attn.o_proj.module'],
            ['mlp.up_proj.module', 'mlp.gate_proj.module'],
            ['mlp.down_proj.module']
        ]
    else:
        sequential = [
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, groupsize=-1, sym=layer_weight_sym, mse=args.w_clip
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                # 构造position_embeddings
                position_embeddings = model.model.rotary_emb(inps[j], position_ids)
                # 注意这里必须显式传入position_embeddings参数（可能因为transformers库版本不同），并且每层都要重新计算，Catcher只捕获第0层的，不能一直用那个
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()

            row_parallel_linears = [
                "self_attn.o_proj",
                "mlp.down_proj",
            ]
            for name in subset:
                if any([row_parallel_linear in name for row_parallel_linear in row_parallel_linears]) and args.w_groupsize == -1:
                    groupsize = subset[name].weight.data.shape[1] // args.tp
                else:
                    groupsize = args.w_groupsize

                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, static_groups=args.act_order
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            # 构造position_embeddings
            position_embeddings = model.model.rotary_emb(inps[j], position_ids)
            # 注意这里必须显式传入position_embeddings参数（可能因为transformers库版本不同），并且每层都要重新计算，Catcher只捕获第0层的，不能一直用那个
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.npu.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers

       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo
    '''
    layers = model.model.layers
    torch.npu.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer,
                                            layers=[torch.nn.Linear])
        row_parallel_linears = [
            "self_attn.o_proj",
            "mlp.down_proj",
        ]

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if any([row_parallel_linear in name for row_parallel_linear in row_parallel_linears]) and args.w_groupsize == -1:
                groupsize = subset[name].weight.data.shape[1] // args.tp
            else:
                groupsize = args.w_groupsize

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, groupsize=groupsize, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.npu.empty_cache()
        del layer
            
    utils.cleanup_memory(verbos=True)
    return quantizers
