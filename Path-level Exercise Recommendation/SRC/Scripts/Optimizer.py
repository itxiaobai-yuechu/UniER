import torch
import torch.nn as nn

class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *data):

        inputs, rewards = data[:-1], data[-1]
        output_data = self.model(*inputs)
        
        return self.criterion(output_data[1], rewards)

    def backup(self, *data):

        inputs, rewards = data[:-1], data[-1]
        output_data = self.model.backup(*inputs)
        return self.criterion(output_data, rewards)


class ModelWithOptimizer(nn.Module):
    def __init__(self, model_with_loss, optimizer):
        super(ModelWithOptimizer, self).__init__()
        self.model_with_loss = model_with_loss
        self.optimizer = optimizer

    def forward(self, *data):

        self.optimizer.zero_grad()
        
        loss = self.model_with_loss.backup(*data)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model_with_loss.parameters(), max_norm=20)
        
        self.optimizer.step()
        
        return loss