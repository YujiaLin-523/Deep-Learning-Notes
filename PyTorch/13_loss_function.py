import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

# L1loss
l1_loss = L1Loss(reduction='sum')
l1_result = l1_loss(input_tensor, target)

# MSELoss
mse_loss = MSELoss()
mse_result = mse_loss(input_tensor, target)

# CELoss
ce_input = torch.tensor([0.1, 0.2, 0.3])
ce_target = torch.tensor([1])
ce_input = torch.reshape(ce_input, (1, 3))
ce_loss = CrossEntropyLoss()
ce_result = ce_loss(ce_input, ce_target)

print(l1_result)
print(mse_result)
print(ce_result)
