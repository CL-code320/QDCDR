import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.singleVBGE import singleVBGE
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from torch.distributions import Distribution


from numbers import Number
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union


class QDCDR(nn.Module):
    def __init__(self, opt):
        super(QDCDR, self).__init__()
        self.opt=opt

        self.source_specific_GNN = singleVBGE(opt)

        self.target_specific_GNN = singleVBGE(opt)

        self.share_GNN = singleVBGE(opt)

        self.dropout = opt["dropout"]

        self.ratio = opt["ratio"]
        # self.average_rate = opt["average_rate"]

        self.source_user_embedding = nn.Embedding(opt["source_user_num"], opt["feature_dim"])
        self.target_user_embedding = nn.Embedding(opt["target_user_num"], opt["feature_dim"])
        self.source_item_embedding = nn.Embedding(opt["source_item_num"], opt["feature_dim"])
        self.target_item_embedding = nn.Embedding(opt["target_item_num"], opt["feature_dim"])
        self.share_user_embedding = nn.Embedding(opt["share_user_num"], opt["feature_dim"])

        self.share_mean = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"]) # Linear(in_features=256, out_features=128, bias=True)
        self.share_sigma = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"]) # Linear(in_features=256, out_features=128, bias=True)


        self.user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.source_user_index = torch.arange(0, self.opt["source_user_num"], 1)
        self.target_user_index = torch.arange(0, self.opt["target_user_num"], 1)
        self.source_item_index = torch.arange(0, self.opt["source_item_num"], 1)
        self.target_item_index = torch.arange(0, self.opt["target_item_num"], 1)
        self.share_item_index = torch.arange(0, self.opt["source_item_num"] + self.opt["target_item_num"], 1)

        if self.opt["cuda"]:
            self.user_index = self.user_index.cuda()
            self.source_user_index = self.source_user_index.cuda()
            self.target_user_index = self.target_user_index.cuda()
            self.source_item_index = self.source_item_index.cuda()
            self.target_item_index = self.target_item_index.cuda()
            self.share_item_index = self.share_item_index.cuda()

        
    def source_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.source_predict_1(fea)
        out = F.relu(out)
        out = self.source_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def target_predict_nn(self, user_embedding, item_embedding):
        fea = torch.cat((user_embedding, item_embedding), dim=-1)
        out = self.target_predict_1(fea)
        out = F.relu(out)
        out = self.target_predict_2(out)
        out = torch.sigmoid(out)
        return out

    def source_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def target_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output

    def share_predict_dot(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # return torch.sigmoid(output)
        return output


    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """Using std to compute KLD"""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        # sigma_1 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_1))
        # sigma_2 = 0.1 + 0.9 * F.softplus(torch.exp(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def reparameters(self, mean, logstd):
        # sigma = 0.1 + 0.9 * F.softplus(torch.exp(logstd))
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.opt["hidden_dim"]).cuda(mean.device)
        if self.share_mean.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.opt["beta"]) * kld_loss
    
    def quantized_disentangle(self,  mu_single, logsigma_single, mu_mix, logsigma_mix):
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_single))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_mix))
        q_single = Normal(mu_single, sigma_1)
        q_mix = Normal(mu_mix, sigma_2)
        share_mask = self.compute_shared_mask_from_posteriors(
            q_single, q_mix, self.ratio
        )
        new_ds_posterior = self.make_shared_posteriors(
            q_single, q_mix, share_mask
        )
        return new_ds_posterior, share_mask

    def compute_shared_mask_from_posteriors(self, d0_posterior: Distribution, d1_posterior: Distribution, ratio=0.5):
        kl_deltas_d1_d0 = kl_divergence(d1_posterior, d0_posterior)
        kl_deltas_d0_d1 = kl_divergence(d0_posterior, d1_posterior)
        z_deltas = (0.5 * kl_deltas_d1_d0) + (0.5 * kl_deltas_d0_d1)
        # print(f"z_deltas min: {z_deltas.min().item()}, max: {z_deltas.max().item()}, mean: {z_deltas.mean().item()}, std: {z_deltas.std().item()}")
        assert 0 <= ratio <= 1, f"ratio must be in the range: 0 <= ratio <= 1, got: {repr(ratio)}"
        # threshold τ
        maximums = z_deltas.max(axis=1, keepdim=True).values  # (B, 1)
        minimums = z_deltas.min(axis=1, keepdim=True).values  # (B, 1)
        z_threshs = torch.lerp(minimums, maximums, weight=ratio)  # (B, 1)
        # true if 'unchanged' and should be average
        shared_mask = z_deltas < z_threshs  # broadcast (B, Z) and (B, 1) -> (B, Z)
        # return
        return shared_mask
        

    def make_shared_posteriors(self, d0_posterior: Normal, d1_posterior: Normal, share_mask: torch.Tensor):
        ave_posterior = self.compute_average_gvae(d0_posterior, d1_posterior)
        ave_d0_posterior = Normal(
            loc=torch.where(share_mask, ave_posterior.loc, d0_posterior.loc),
            scale=torch.where(share_mask, ave_posterior.scale, d0_posterior.scale),
        )
        ave_d1_posterior = Normal(
            loc=torch.where(share_mask, ave_posterior.loc, d1_posterior.loc),
            scale=torch.where(share_mask, ave_posterior.scale, d1_posterior.scale),
        )
        # return values
        return ave_d0_posterior, ave_d1_posterior


    def compute_average_gvae(self, d0_posterior: Normal, d1_posterior: Normal) -> Normal:
        """
        Compute the arithmetic mean of the encoder distributions.
        """
        assert isinstance(
            d0_posterior, Normal
        ), f"posterior distributions must be {Normal.__name__} distributions, got: {type(d0_posterior)}"
        assert isinstance(
            d1_posterior, Normal
        ), f"posterior distributions must be {Normal.__name__} distributions, got: {type(d1_posterior)}"
        # averages
        # ave_var = 0.5 * (d0_posterior.variance + d1_posterior.variance)
        # ave_var = 0.1 * d0_posterior.variance + 0.9 * d1_posterior.variance
        ave_var = d1_posterior.variance
        # ave_mean = 0.5 * (d0_posterior.mean + d1_posterior.mean)
        # ave_mean = 0.1 * d0_posterior.mean + 0.9 * d1_posterior.mean
        ave_mean = d1_posterior.mean
        # done!
        return Normal(loc=ave_mean, scale=torch.sqrt(ave_var))
    
    
    
    def compute_ave_reg_loss(
        self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution], zs_sampled: Sequence[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        kl_loss = compute_ave_loss(compute_kl_loss, ds_posterior, ds_prior, zs_sampled)
        return kl_loss
        
    def compute_average_true_false(self, shared_mask: torch.Tensor):
        """
        Calculate the average number of True (shared samples) and False (non-shared samples) per user in shared_mask.

        Args:
            shared_mask (torch.Tensor): Boolean tensor of shape (B, Z), where each user corresponds to a True/False vector.

        Returns:
            avg_true (float): Average number of True values
            avg_false (float): Average number of False values
        """
        # Calculate the number of True values for each user
        num_true_per_user = shared_mask.sum(dim=1)  # (B,)

        # Calculate the number of False values for each user
        num_false_per_user = (~shared_mask).sum(dim=1)  # (B,)

        # Calculate the averages
        avg_true = num_true_per_user.float().mean().item()
        avg_false = num_false_per_user.float().mean().item()

        return avg_true, avg_false

    def forward(self, source_UV, source_VU, target_UV, target_VU, share_UV, share_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        share_user = self.share_user_embedding(self.user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        share_item = torch.cat([source_item, target_item], dim=0)  #  [source_item_num + target_item_num, feature_dim]

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item, source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item, target_UV, target_VU)
        share_learn_specific_user, share_learn_specific_item = self.share_GNN(share_user, share_item, share_UV, share_VU)

        source_user_mean, source_user_sigma = self.source_specific_GNN.forward_user_share(source_user, source_UV, source_VU)
        target_user_mean, target_user_sigma = self.target_specific_GNN.forward_user_share(target_user, target_UV, target_VU)
        share_user_mean, share_user_sigma = self.share_GNN.forward_user_share(share_user, share_UV, share_VU)


        #使用量化解耦
        prior = (Normal(torch.zeros_like(share_user_mean), torch.ones_like(share_user_sigma)), Normal(torch.zeros_like(share_user_mean), torch.ones_like(share_user_sigma)))


        new_ds_posterior_source, share_mask_source = self.quantized_disentangle(source_user_mean, source_user_sigma, share_user_mean, share_user_sigma)
        zs_sampled_source = tuple(d.rsample() for d in new_ds_posterior_source)
        eg_loss_source = self.compute_ave_reg_loss(new_ds_posterior_source, prior, zs_sampled_source)
        zs_stack_source = torch.stack(zs_sampled_source)
        average_zs_source = torch.mean(zs_stack_source, dim=0)
        user_share_source = torch.where(share_mask_source, average_zs_source, torch.zeros_like(average_zs_source))


        new_ds_posterior_target, share_mask_target = self.quantized_disentangle(target_user_mean, target_user_sigma, share_user_mean, share_user_sigma)
        zs_sampled_target = tuple(d.rsample() for d in new_ds_posterior_target)
        eg_loss_target = self.compute_ave_reg_loss(new_ds_posterior_target, prior, zs_sampled_target)
        zs_stack_target = torch.stack(zs_sampled_target)
        average_zs_target = torch.mean(zs_stack_target, dim=0)
        user_share_target = torch.where(share_mask_target, average_zs_target, torch.zeros_like(average_zs_target))

        self.kld_loss =  self.opt["beta"] * eg_loss_source + self.opt["beta"] * eg_loss_target

        source_learn_user = user_share_target + source_learn_specific_user
        target_learn_user = user_share_source + target_learn_specific_user

        share_mask_dict = {}
        share_mask_dict["source"] = share_mask_source
        share_mask_dict["target"] = share_mask_target
        return source_learn_user, source_learn_specific_item, target_learn_user, target_learn_specific_item, share_learn_specific_user, share_learn_specific_item, share_mask_dict

    def wramup(self, source_UV, source_VU, target_UV, target_VU, share_UV, share_VU):
        source_user = self.source_user_embedding(self.source_user_index)
        target_user = self.target_user_embedding(self.target_user_index)
        source_item = self.source_item_embedding(self.source_item_index)
        target_item = self.target_item_embedding(self.target_item_index)
        share_user = self.share_user_embedding(self.user_index)
        share_item = torch.cat([source_item, target_item], dim=0)  #  [source_item_num + target_item_num, feature_dim]

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(source_user, source_item,
                                                                                          source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(target_user, target_item,
                                                                                          target_UV, target_VU)
        share_learn_specific_user, share_learn_specific_item = self.share_GNN(share_user, share_item, share_UV, share_VU)


        source_user_mean, source_user_sigma = self.source_specific_GNN.forward_user_share(source_user, source_UV, source_VU)
        target_user_mean, target_user_sigma = self.target_specific_GNN.forward_user_share(target_user, target_UV, target_VU)
        share_user_mean, share_user_sigma = self.share_GNN.forward_user_share(share_user, share_UV, share_VU)
        new_ds_posterior_source, share_mask_source = self.quantized_disentangle(source_user_mean, source_user_sigma, share_user_mean, share_user_sigma)
        new_ds_posterior_target, share_mask_target = self.quantized_disentangle(target_user_mean, target_user_sigma, share_user_mean, share_user_sigma)
        share_mask_dict = {}
        share_mask_dict["source"] = share_mask_source
        share_mask_dict["target"] = share_mask_target

        self.kld_loss = 0
        return source_learn_specific_user, source_learn_specific_item, target_learn_specific_user, target_learn_specific_item, share_learn_specific_user, share_learn_specific_item, share_mask_dict


def map_all(fn, *arg_lists, starmap: bool = True, collect_returned: bool = False, common_kwargs: dict = None):
    assert arg_lists, "an empty list of args was passed"
    # check all lengths are the same
    num = len(arg_lists[0])
    assert num > 0
    assert all(len(items) == num for items in arg_lists)
    # update kwargs
    if common_kwargs is None:
        common_kwargs = {}
    # map everything
    if starmap:
        results = (fn(*args, **common_kwargs) for args in zip(*arg_lists))
    else:
        results = (fn(args, **common_kwargs) for args in zip(*arg_lists))
    # zip everything
    if collect_returned:
        return tuple(zip(*results))
    else:
        return tuple(results)

def compute_ave_loss(loss_fn, *arg_list, **common_kwargs) -> torch.Tensor:
    # compute all losses
    losses = map_all(loss_fn, *arg_list, collect_returned=False, common_kwargs=common_kwargs)
    # compute mean loss
    loss = torch.stack(losses).mean(dim=0)
    # return!
    return loss

def compute_kl_loss(
        posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the kl divergence
        """
        kl = torch.distributions.kl_divergence(posterior, prior)
        kl = kl.mean()
        return kl