r""" Provides functions that builds/manipulates correlation tensors """
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class Correlation:
    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []            
        for idx, (query_feat, support_feat, query_feat_uni, support_feat_uni, query_feat_org, support_feat_org) in enumerate(zip(query_feats, support_feats, query_feats_uni, support_feats_uni, query_feats_org, support_feats_org)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)

            corrs.append(corr)          
        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]