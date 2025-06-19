import torch
import torch.nn as nn
import torch.nn.functional as F

class NT_Xent(nn.Module):
    def __init__(self, temp):
        super(NT_Xent, self).__init__()
        self.temperature = temp
        self.mask = None

        
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        self.mask = mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # dimension of similarity matrix
        batch_size = z_i.size(0)


        N = 2 * batch_size
        if self.mask is None or self.batch_size != batch_size:
            self.batch_size = batch_size
            self.mask_correlated_samples()

        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).long()
        # the logits are NxN (N+1?) predictions for imaginary classes.
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class NTXentLoss(nn.Module):
    def __init__(self, temp=0.5):
        super(NTXentLoss, self).__init__()
        self.temp = temp

    def forward(self, z_i, z_j):
        """
        z_i: Tensor of shape [N, D] - first view embeddings
        z_j: Tensor of shape [N, D] - second view embeddings
        Returns:
            Scalar NT-Xent loss
        """
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Normalize embeddings
        z = nn.functional.normalize(z, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T)  # [2N, 2N]
        sim_matrix = sim_matrix / self.temp

        # Mask to remove self-similarity
        mask = (~torch.eye(2 * N, 2 * N, dtype=bool, device=z.device)).float()

        # Numerator: positive pairs (i, j)
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / self.temp)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # [2N]

        # Denominator: sum over all except self
        denom = torch.sum(torch.exp(sim_matrix) * mask, dim=1)  # [2N]

        loss = -torch.log(pos_sim / denom)
        return loss.mean()





class QuaternionLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super(QuaternionLoss, self).__init__()
        assert loss_type in ['mse', 'relative'], "loss_type must be 'mse' or 'relative'"
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss_fn = self.quaternion_mse_loss
        else:
            self.loss_fn = self.quaternion_relative_loss

    def forward(self, q1_pred, q2_pred, q1_true, q2_true):
        return self.loss_fn(q1_pred, q2_pred, q1_true, q2_true)

    def euler_to_quaternion(self, euler: torch.Tensor, order: str = 'xyz') -> torch.Tensor:
        roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack((w, x, y, z), dim=-1)

    def normalize_quaternion(self, q: torch.Tensor, eps=1e-8) -> torch.Tensor:
        return q / (q.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))

    def quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        w = q[..., 0:1]
        xyz = -q[..., 1:]
        return torch.cat([w, xyz], dim=-1)

    def quaternion_multiply(self, q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = r.unbind(-1)

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack((w, x, y, z), dim=-1)

    def quaternion_relative_loss(self, q1_pred, q2_pred, q1_true, q2_true):
        q1_pred = self.normalize_quaternion(q1_pred)
        q2_pred = self.normalize_quaternion(q2_pred)
        q1_true = self.normalize_quaternion(q1_true)
        q2_true = self.normalize_quaternion(q2_true)

        q1_pred_inv = self.quaternion_conjugate(q1_pred)
        q1_true_inv = self.quaternion_conjugate(q1_true)

        rel_pred = self.quaternion_multiply(q2_pred, q1_pred_inv)
        rel_true = self.quaternion_multiply(q2_true, q1_true_inv)

        dot = torch.abs(torch.sum(rel_pred * rel_true, dim=-1))
        
        #loss = 1.0 - dot
        loss = (2 * torch.acos(dot)) ** 2
        return loss.mean()

    def quaternion_mse_loss(self, q1_pred, q2_pred, q1_true, q2_true):
        q1_pred = self.normalize_quaternion(q1_pred)
        q2_pred = self.normalize_quaternion(q2_pred)
        q1_true = self.normalize_quaternion(q1_true)
        q2_true = self.normalize_quaternion(q2_true)

        dot1 = torch.sum(q1_pred * q1_true, dim=-1)
        dot1 = torch.clamp(dot1, -1.0, 1.0)

        dot2 = torch.sum(q2_pred * q2_true, dim=-1)
        dot2 = torch.clamp(dot2, -1.0, 1.0)

        dots = torch.cat([dot1, dot2], dim=0)
        return F.mse_loss(dots, torch.ones_like(dots))
