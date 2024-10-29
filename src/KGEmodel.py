# this file is based on https://github.com/hughxiouge/CompoundE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KGEModel(nn.Module):
    def __init__(self, model_name, gamma, hidden_dim):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'PairRE': self.PairRE,
            'RotatEv2': self.RotatEv2,
            'CompoundE': self.CompoundE
        }

    def forward(self, sample, mode='positive'):
        if mode == 'positive':
            head, relation, tail = sample
        elif mode == 'negative':
            head, relation, tail = sample
            # batch_size, negative_sample_size = tail.size(0), tail.size(1)
            # tail = tail.view(batch_size, negative_sample_size, -1)
            negative_sample_size, tail_numb = tail.size(0), tail.size(1)
            tail = tail.view(negative_sample_size, tail_numb, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        if self.model_name in self.model_func:
            score = self.model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        # score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        if mode == 'positive':
            ke_head = tail - relation
            return score, ke_head
        else:
            return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):

        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        # score = score.sum(dim=2)
        score = score.sum(dim=-1)
        if mode == 'positive':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail

            ke_head = torch.cat((re_score, im_score), dim=-1)
            return score, ke_head
        else:
            return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        # score = self.gamma.item() - score.sum(dim=2)
        score = self.gamma.item() - score.sum(dim=-1)
        if mode == 'positive':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail

            ke_head = torch.cat((re_score, im_score), dim=-1)
            return score, ke_head
        else:
            return score

    def RotatEv2(self, head, relation, tail, mode, r_norm=None):
        pi = 3.14159265358979323846

        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        # re_relation_head, re_relation_tail = torch.chunk(re_relation, 2, dim=2)
        # im_relation_head, im_relation_tail = torch.chunk(im_relation, 2, dim=2)
        re_relation_head, re_relation_tail = torch.chunk(re_relation, 2, dim=-1)
        im_relation_head, im_relation_tail = torch.chunk(im_relation, 2, dim=-1)

        re_score_head = re_head * re_relation_head - im_head * im_relation_head
        im_score_head = re_head * im_relation_head + im_head * re_relation_head

        re_score_tail = re_tail * re_relation_tail - im_tail * im_relation_tail
        im_score_tail = re_tail * im_relation_tail + im_tail * re_relation_tail

        re_score = re_score_head - re_score_tail
        im_score = im_score_head - im_score_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        # score = self.gamma.item() - score.sum(dim=2)
        score = self.gamma.item() - score.sum(dim=-1)
        return score

    def PairRE(self, head, relation, tail, mode):
        # re_head, re_tail = torch.chunk(relation, 2, dim=2)
        re_head, re_tail = torch.chunk(relation, 2, dim=-1)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        # score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        if mode == 'positive':
            ke_head = tail * re_tail / re_head
            return score, ke_head
        else:
            return score

    def CompoundE(self, head, relation, tail, mode):
        # tail_scale, tail_translate, theta = torch.chunk(relation, 3, dim=2)
        # theta, _ = torch.chunk(theta, 2, dim=2)
        tail_scale, tail_translate, theta = torch.chunk(relation, 3, dim=-1)  # (2,512)
        theta, _ = torch.chunk(theta, 2, dim=-1)  # (2,256)

        head = F.normalize(head, 2, -1)  # (512)
        tail = F.normalize(tail, 2, -1)  # (2,512)

        pi = 3.14159265358979323846

        theta = theta / (self.embedding_range.item() / pi)  # (2,256)

        re_rotation = torch.cos(theta)  # (2,256)
        im_rotation = torch.sin(theta)

        re_rotation = re_rotation.unsqueeze(-1)  # (2,256,1)
        im_rotation = im_rotation.unsqueeze(-1)

        # tail = tail.view((tail.shape[0], tail.shape[1], -1, 2))

        if mode == 'positive':
            tail = tail.view((tail.shape[0], -1, 2))  # (2,256,2)
            tail_r = torch.cat((re_rotation * tail[:, :, 0:1], im_rotation * tail[:, :, 0:1]), dim=-1)
            tail_r += torch.cat((-im_rotation * tail[:, :, 1:], re_rotation * tail[:, :, 1:]), dim=-1)
            tail_r = tail_r.view((tail_r.shape[0], -1))  # (2,512)# (2,256,2)
        elif mode == 'negative':
            tail = tail.view((tail.shape[0], tail.shape[1], -1, 2))  # (12,28,256,2).
            tail_r = torch.cat((re_rotation * tail[:, :, :, 0:1], im_rotation * tail[:, :, :, 0:1]), dim=-1)
            tail_r += torch.cat((-im_rotation * tail[:, :, :, 1:], re_rotation * tail[:, :, :, 1:]), dim=-1)
            tail_r = tail_r.view((tail_r.shape[0], tail_r.shape[1], -1))

        tail_r += tail_translate
        tail_r *= tail_scale

        score = head - tail_r
        # score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        # return score
        if mode == 'positive':
            ke_head = tail_r
            return score, ke_head
        else:
            return score

