import torch
import torch.nn as nn

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
