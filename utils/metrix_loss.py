import numpy as np
import torch
import logging
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
import random
import itertools

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f 

def margin(x, y):
	return x - y

def mixup_data(p, n, alpha=0.4):    
	lam = np.random.beta(alpha, alpha)
	mixed_x = lam * p + (1 - lam) * n

	return mixed_x, lam

def loss_pos_mixup(sim_mix, a1, pos):
	
	pos_pair_dist = []
	if len(a1) > 0:
		pos_pair_dist = torch.diagonal(sim_mix, 0)

	pos_pairs = (a1, pos)

	if len(pos_pair_dist) > 0:
		pos_loss = torch.nn.functional.relu(margin(pos_pair_dist, 0.0))

	loss_pos_dict =  {
		"pos_loss": {
			"losses": pos_loss,
			"indices": pos_pairs,
			"reduction_type": "pos_pair",
		}
	}

	return loss_pos_dict

def loss_posneg_mixup(sim_mix, pos, a2, neg, lam):
		
	if len(a2) > 0:
		pair_dist = torch.diagonal(sim_mix, 0)

	indices_tuple = (a2, pos, a2, neg)
	pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
	neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)
	
	if len(pair_dist) > 0:
		pos_loss = lam*torch.nn.functional.relu(margin(pair_dist, 0.0))
	if len(pair_dist) > 0:
		neg_loss = (1-lam)*torch.nn.functional.relu(margin(0.5, pair_dist))


	loss_pos_dict =  {
		"pos_loss": {
			"losses": pos_loss,
			"indices": pos_pairs,
			"reduction_type": "pos_pair",
		}
	}

	loss_neg_dict =  {
		"neg_loss": {
			"losses": neg_loss,
			"indices": neg_pairs,
			"reduction_type": "neg_pair",
		},
	}

	return loss_pos_dict, loss_neg_dict


def multisimilarity_positive_loss(sim_mix, alpha, base, a1, p,distance, clean_embedding, a_mix, p_mix,lam):
	
	# ---------clean ----------------
	# clean similarity matrix
	mat = distance(clean_embedding)
	# positive part margin
	pos_exp = alpha*distance.margin(mat, base)
	
	# empty martix having similarity matrix shape
	pos_mask = torch.zeros_like(mat)
	# Put one to elements of pos mask that correspond to positive pairs
	pos_mask[a1, p] = 1

	# True -> Negatives
	# False -> Positives
	pos_exp = pos_exp.masked_fill(~pos_mask.bool(), c_f.neg_inf(pos_exp.dtype))
	
	# zeros has shape torch.Size([100, 1])
	zeros = torch.zeros(pos_exp.size(1 - 1), dtype=pos_exp.dtype, device=pos_exp.device).unsqueeze(1)
	# pos_exp has shape torch.Size([100, 101])
	pos_exp = torch.cat([pos_exp, zeros], dim=1)

	# ----------mix ---------------------------------
	
	pos_exp_mix =  alpha*distance.margin(sim_mix, base)
	pos_mask_mix = torch.zeros_like(sim_mix)
	# pos_mask_mix = lam*torch.eye(pos_mask_mix.shape[0]).cuda()
	pos_mask_mix[a_mix, p_mix] = lam

	pos_exp_mix = pos_exp_mix.masked_fill(~pos_mask_mix.bool(), c_f.neg_inf(pos_exp_mix.dtype))
	zeros_mix = torch.zeros(pos_exp_mix.size(1 - 1), dtype=pos_exp_mix.dtype, device=pos_exp_mix.device).unsqueeze(1)
	pos_exp_mix = torch.cat([pos_exp_mix, zeros_mix], dim=1)
	
	# -----------------------------------------------

	# Clean one
	possumexp = lam*torch.sum(torch.exp(pos_exp), dim=1, keepdim=True) 
	possumexp = possumexp.masked_fill_(~torch.any(pos_mask.bool(), dim=1, keepdim=True), 0)	

	# Mix one
	possumexp_mix = torch.sum(torch.exp(pos_exp_mix), dim=1, keepdim=True) 
	possumexp_mix = possumexp_mix.masked_fill_(~torch.any(pos_mask_mix.bool(), dim=1, keepdim=True), 0)	

	zeros_pad = torch.zeros(possumexp_mix.shape[0] - possumexp.shape[0]).cuda()
	
	posLogsumexp = torch.log(torch.cat([possumexp.squeeze(1), zeros_pad]).unsqueeze(1) + 0.4*possumexp_mix)
	
	idxs,_ = torch.where(posLogsumexp < -100000)
	posLogsumexp[idxs] = 0
	
	pos_loss = (1.0 / alpha) * posLogsumexp
	# exit()

	return pos_loss


def multisimilarity_negative_loss(sim_mix, beta, base, a2, n, distance, clean_embedding, a_mix, n_mix, lam):
	
	# ---------clean ----------------
	mat = distance(clean_embedding)
	neg_exp = beta * distance.margin(base, mat)

	neg_mask = torch.zeros_like(mat)
	neg_mask[a2,n] = 1

	neg_exp = neg_exp.masked_fill(~neg_mask.bool(), c_f.neg_inf(neg_exp.dtype))
	zeros = torch.zeros(neg_exp.size(1 - 1), dtype=neg_exp.dtype, device=neg_exp.device).unsqueeze(1)
	neg_exp = torch.cat([neg_exp, zeros], dim=1)

	# ----------mix ---------------------------------
	
	neg_exp_mix =  beta * distance.margin(base, sim_mix)
	neg_mask_mix = torch.zeros_like(sim_mix)
	# neg_mask_mix = (1-lam)*torch.eye(neg_mask_mix.shape[0]).cuda()
	neg_mask[a_mix,n_mix] = (1.0-lam)

	neg_exp_mix = neg_exp_mix.masked_fill(~neg_mask_mix.bool(), c_f.neg_inf(neg_exp_mix.dtype))
	zeros_mix = torch.zeros(neg_exp_mix.size(1 - 1), dtype=neg_exp_mix.dtype, device=neg_exp_mix.device).unsqueeze(1)
	neg_exp_mix = torch.cat([neg_exp_mix, zeros_mix], dim=1)
	
	# -----------------------------------------------

	negsumexp = (1-lam)*torch.sum(torch.exp(neg_exp), dim=1, keepdim=True)
	negsumexp = negsumexp.masked_fill(~torch.any(neg_mask.bool(), dim=1, keepdim=True), 0)

	negsumexp_mix = torch.sum(torch.exp(neg_exp_mix), dim=1, keepdim=True)
	negsumexp_mix = negsumexp_mix.masked_fill(~torch.any(neg_mask_mix.bool(), dim=1, keepdim=True), 0)
	
	zeros_pad = torch.zeros(negsumexp_mix.shape[0] - negsumexp.shape[0]).cuda()

	negLogsumexp = torch.log(torch.cat([negsumexp.squeeze(1), zeros_pad]).unsqueeze(1)  + 0.4*negsumexp_mix)
	
	idxs,_ = torch.where(negLogsumexp < -100000) # remove the -Inf due to the log
	negLogsumexp[idxs] = 0

	neg_loss = (1.0 / beta) * negLogsumexp
	

	return neg_loss


def embed_posneg_pair_mixup_for_pos_anchor(a1, pos, a2, neg, embedding, target, sim_mat):
	new_anc, new_neg, new_pos = [],[],[]
	for anchor_pos in a1:

		idxs = torch.where(anchor_pos == a2)[0]
		new_neg.append(neg[idxs].cpu().numpy()) 
		new_anc.append(a2[idxs].cpu().numpy())

		index_pos = torch.where(anchor_pos == a1)[0]

		new_pos.append([pos[index_pos].cpu().numpy()[0]]*len(idxs))
		
	a2 = list(itertools.chain.from_iterable(new_anc))
	neg = list(itertools.chain.from_iterable(new_neg))
	pos = list(itertools.chain.from_iterable(new_pos))

	anchor_embedding = embedding[a2]
	pos_embedding = embedding[pos]
	neg_embedding = embedding[neg]

	target_pos = target[pos]
	target_neg = target[neg]

	mixed_embedding, lam = mixup_data(pos_embedding, neg_embedding, alpha=2.0)

	similarity_mixed = sim_mat(anchor_embedding, mixed_embedding)

	return similarity_mixed, target_pos, target_neg, torch.from_numpy(np.array(a2)), torch.from_numpy(np.array(pos)), torch.from_numpy(np.array(neg)), lam