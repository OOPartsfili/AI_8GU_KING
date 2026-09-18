"""教学实现的数值、梯度与行为校验：python test_transformer_handwrite.py。"""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from transformer_handwrite import (attention, split_heads, merge_heads, causal_allowed,
    padding_allowed, MultiHeadAttention, LayerNorm, RMSNorm, FeedForward, SwiGLU,
    sinusoidal_positions, TinyLanguageModel, TinySeq2Seq, next_token_loss,
    generate, apply_rope, expand_kv, learning_demo)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)

    def close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_split_merge_preserves_token_order(self):
        x = torch.arange(48).reshape(2, 3, 8)
        heads = split_heads(x, 2)
        self.assertEqual(tuple(heads.shape), (2, 2, 3, 4))
        self.close(heads[0, 1, 1], x[0, 1, 4:])
        self.close(merge_heads(heads), x)

    def test_attention_weighted_sum(self):
        q = torch.tensor([[[[1., 0.]]]])
        k = torch.tensor([[[[1., 0.], [0., 1.]]]])
        v = torch.tensor([[[[2., 0.], [0., 4.]]]])
        y, weights = attention(q, k, v)
        a = torch.sigmoid(torch.tensor(1 / 2**0.5))
        self.close(y.flatten(), torch.stack([2*a, 4*(1-a)]))
        self.close(weights.sum(-1), torch.ones(1,1,1))

    def test_attention_matches_sdpa_forward_and_gradient(self):
        q, k, v = [torch.randn(2, 2, 3, 4, dtype=torch.double, requires_grad=True) for _ in range(3)]
        mask = causal_allowed(3, 3)
        actual, _ = attention(q, k, v, mask)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        self.close(actual, expected)
        a = torch.autograd.grad(actual.square().sum(), (q,k,v), retain_graph=True)
        b = torch.autograd.grad(expected.square().sum(), (q,k,v))
        for x,y in zip(a,b): self.close(x,y)

    def test_gradient_matches_finite_difference(self):
        args = tuple(torch.randn(1,1,2,2, dtype=torch.double, requires_grad=True) for _ in range(3))
        self.assertTrue(torch.autograd.gradcheck(lambda q,k,v: attention(q,k,v)[0], args))

    def test_mask_all_hidden_fails_explicitly(self):
        x=torch.randn(1,1,2,2)
        with self.assertRaisesRegex(ValueError, '任何 key'):
            attention(x,x,x,torch.zeros(2,2,dtype=torch.bool))

    def test_padding_cannot_affect_valid_queries(self):
        q,k,v=[torch.randn(2,2,4,3) for _ in range(3)]
        mask=padding_allowed(torch.tensor([[True,True,False,False]]*2))
        expected,_=attention(q,k,v,mask)
        k[:,:,2:]+=1000; v[:,:,2:]-=1000
        self.close(attention(q,k,v,mask)[0],expected)

    def test_mha_matches_pytorch_self_and_cross(self):
        manual=MultiHeadAttention(8,2).double().eval()
        reference=nn.MultiheadAttention(8,2,batch_first=True).double().eval()
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([manual.q_proj.weight,manual.k_proj.weight,manual.v_proj.weight]))
            reference.in_proj_bias.copy_(torch.cat([manual.q_proj.bias,manual.k_proj.bias,manual.v_proj.bias]))
            reference.out_proj.load_state_dict(manual.out_proj.state_dict())
        x=torch.randn(2,3,8,dtype=torch.double)
        for context in (None,torch.randn(2,5,8,dtype=torch.double)):
            src=x if context is None else context
            mask=causal_allowed(3,3) if context is None else torch.ones(3,5,dtype=torch.bool)
            y,w,_=manual(x,context=context,allowed=mask)
            # nn.MultiheadAttention 的 True 表示禁止，恰与本教程相反。
            yr,wr=reference(x,src,src,attn_mask=~mask,average_attn_weights=False)
            self.close(y,yr); self.close(w,wr)

    def test_norm_reference_and_per_token_independence(self):
        x=torch.randn(2,3,8,dtype=torch.double,requires_grad=True)
        norm=LayerNorm(8).double()
        self.close(norm(x),F.layer_norm(x,(8,),norm.weight,norm.bias,norm.eps))
        changed=x.detach().clone(); changed[:,2]+=99
        self.close(norm(changed)[:,:2],norm(x)[:,:2])
        self.close(RMSNorm(8).double()(x), F.rms_norm(x,(8,),eps=1e-6))

    def test_ffn_is_positionwise_and_nonlinear(self):
        x=torch.randn(2,3,8)
        for layer in (FeedForward(8,16), SwiGLU(8,16)):
            changed=x.clone(); changed[:,2]+=3
            self.close(layer(changed)[:,:2],layer(x)[:,:2])
            self.assertEqual(layer(x).shape,x.shape)

    def test_sinusoidal_zero_position_and_distinct_rows(self):
        p=sinusoidal_positions(4,8)
        self.close(p[0,0::2],torch.zeros(4)); self.close(p[0,1::2],torch.ones(4))
        self.assertFalse(torch.equal(p[0],p[1]))

    def test_lm_future_tokens_do_not_change_past_logits(self):
        model=TinyLanguageModel().eval()
        ids=torch.tensor([[0,1,2,3,4]])
        changed=ids.clone(); changed[:,3:]=7
        self.close(model(ids)[0][:,:3],model(changed)[0][:,:3])

    def test_cache_chunk_and_single_token_equal_full_forward(self):
        model=TinyLanguageModel().eval()
        ids=torch.tensor([[0,1,2,3,4,5]])
        full,_=model(ids)
        with torch.no_grad():
            prefix,caches=model(ids[:,:3],use_cache=True)
            chunk,caches=model(ids[:,3:5],caches=caches,use_cache=True)
            last,caches=model(ids[:,5:],caches=caches,use_cache=True)
        self.close(torch.cat([prefix,chunk,last],1),full)
        self.assertTrue(all(c[0].size(-2)==6 for c in caches))
        self.assertEqual(causal_allowed(2,5,3).tolist(),[[True]*4+[False],[True]*5])

    def test_cached_generation_equal_and_mode_restored(self):
        model=TinyLanguageModel()
        ids=torch.tensor([[0,1,2]])
        self.close(generate(model,ids,5,True),generate(model,ids,5,False))
        self.assertTrue(model.training)

    def test_loss_shift_matches_explicit_log_softmax(self):
        model=TinyLanguageModel()
        ids=torch.tensor([[0,1,2,3]])
        logp=model(ids[:,:-1])[0].log_softmax(-1)
        expected=-logp.gather(-1,ids[:,1:,None]).mean()
        self.close(next_token_loss(model,ids),expected)

    def test_seq2seq_padding_and_causality(self):
        model=TinySeq2Seq().eval()
        src=torch.tensor([[1,2,3,0]]); valid=torch.tensor([[True,True,True,False]])
        target=torch.tensor([[1,4,5]])
        y=model(src,target,valid)
        changed_src=src.clone(); changed_src[:,-1]=10
        self.close(model(changed_src,target,valid),y)
        changed_target=target.clone(); changed_target[:,-1]=11
        self.close(model(src,changed_target,valid)[:,:2],y[:,:2])
        model(src,target,valid).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_rope_preserves_norm_and_relative_dot(self):
        q,k=[torch.randn(1,1,1,8,dtype=torch.double) for _ in range(2)]
        qm=apply_rope(q,torch.tensor([3])); kn=apply_rope(k,torch.tensor([5]))
        self.close(qm.square().sum(-1),q.square().sum(-1))
        self.close((qm*kn).sum(-1),(q*apply_rope(k,torch.tensor([2]))).sum(-1))

    def test_gqa_group_mapping(self):
        k=torch.arange(8.).reshape(1,2,2,2); v=k+1
        ke,ve=expand_kv(k,v,4)
        self.close(ke[:,0],k[:,0]); self.close(ke[:,1],k[:,0])
        self.close(ke[:,2],k[:,1]); self.close(ve[:,3],v[:,1])
        with self.assertRaises(ValueError): expand_kv(k,v,3)

    def test_layers_have_independent_parameters_and_gradients(self):
        model=TinyLanguageModel()
        self.assertIsNot(model.blocks[0].attn.q_proj.weight,model.blocks[1].attn.q_proj.weight)
        next_token_loss(model,torch.tensor([[0,1,2,3]])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_toy_learning_reduces_loss_and_generates_pattern(self):
        r=learning_demo()
        self.assertLess(r['last_loss'],r['first_loss']*0.1)
        self.assertEqual(r['generated'],r['expected'])


if __name__=='__main__':
    unittest.main(verbosity=2)
