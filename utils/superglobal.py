# written by Shihao Shao (shaoshihao@pku.edu.cn)


import torch
from torch import nn

class MDescAug(nn.Module):
    """ Top-M Descriptor Augmentation"""
    def __init__(self, M = 30, K = 19, beta = 0.15):
        super(MDescAug, self).__init__()
        self.M = M
        self.K = K + 1
        self.beta = beta

    def forward(self, X, Q, ranks):
        
        ranks_trans_1000 = torch.transpose(ranks,1,0)[:,:self.M] # 70 400 
        
        
        X_tensor1 = X[ranks_trans_1000].clone().detach().cuda()
        
        res_ie = torch.einsum('abc,adc->abd',
                X_tensor1, X_tensor1) # 70 400 400

        del X_tensor1

        res_ie_ranks = torch.unsqueeze(torch.argsort(-res_ie.clone(), axis=-1)[:,:,:self.K],-1) # 70 400 10 1
        res_ie_ranks_value = torch.unsqueeze(-torch.sort(-res_ie.clone(), axis=-1)[0][:,:,:self.K],-1) # 70 400 10 1
        del res_ie

        res_ie_ranks_value[:,:,1:,:] *= self.beta
        res_ie_ranks_value[:,:,0:1,:] = 1.
        res_ie_ranks = torch.squeeze(res_ie_ranks,-1) # 70 400 10
        x_dba = X[ranks_trans_1000] # 70 1 400 2048

        del X
        
        
        x_dba_list = []
        for i,j in zip(res_ie_ranks,x_dba):
            # we should avoid for-loop in python, 
            # thus even make the numbers in paper look nicer, 
            # but i just want to go to bed.
            # i 400 10 j # 400 2048
            x_dba_list.append(j[i])

        del res_ie_ranks
        
        x_dba = torch.stack(x_dba_list,0) # 70 400 10 2048
        
        x_dba = torch.sum(x_dba * res_ie_ranks_value, 2) / torch.sum(res_ie_ranks_value,2) # 70 400 2048
        del res_ie_ranks_value

        res_top1000_dba = torch.einsum('ac,adc->ad', Q, x_dba) # 70 400 

        del Q
 
        ranks_trans_1000_pre = torch.argsort(-res_top1000_dba,-1) # 70 400
        rerank_dba_final = []
        for i in range(ranks_trans_1000_pre.shape[0]):
            temp_concat = ranks_trans_1000[i][ranks_trans_1000_pre[i]]
            rerank_dba_final.append(temp_concat) # 400
        return rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba
    
class RerankwMDA(nn.Module):
    """ Reranking with maximum descriptors aggregation """
    def __init__(self, M=30, K = 19, beta = 0.15):
        super(RerankwMDA, self).__init__()
        self.M = M 
        self.K = K + 1 # including oneself
        self.beta = beta
    def forward(self, ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba):

        ranks_trans_1000 = torch.stack(rerank_dba_final,0) # 70 400
        ranks_value_trans_1000 = -torch.sort(-res_top1000_dba,-1)[0] # 70 400
        

        ranks_trans = torch.unsqueeze(ranks_trans_1000_pre[:,:self.K],-1) # 70 10 1
        ranks_value_trans = torch.unsqueeze(ranks_value_trans_1000[:,:self.K].clone(),-1) # 70 10 1
        ranks_value_trans[:,:,:] *=self.beta
        
        X1 =torch.take_along_dim(x_dba, ranks_trans,1) # 70 10 2048
        X2 =torch.take_along_dim(x_dba, torch.unsqueeze(ranks_trans_1000_pre,-1),1) # 70 400 2048
        X1 = torch.max(X1, 1, True)[0] # 70 1 2048
        res_rerank = torch.sum(torch.einsum(
            'abc,adc->abd',X1,X2),1) # 70 400
        

        res_rerank = (ranks_value_trans_1000 + res_rerank) / 2. # 70 400
        res_rerank_ranks = torch.argsort(-res_rerank, axis=-1) # 70 400
        
        rerank_qe_final = []
        ranks_transpose = torch.transpose(ranks,1,0)[:,self.M:] # 70 6322-400
        for i in range(res_rerank_ranks.shape[0]):
            temp_concat = torch.concat([ranks_trans_1000[i][res_rerank_ranks[i]],ranks_transpose[i]],0)
            rerank_qe_final.append(temp_concat) # 6322
        ranks = torch.transpose(torch.stack(rerank_qe_final,0),1,0) # 70 6322
        
        return ranks