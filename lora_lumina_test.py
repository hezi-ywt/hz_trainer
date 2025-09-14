from models.LyCORIS.lycoris import create_lycoris, LycorisNetwork
import torch
import torch.nn as nn

LycorisNetwork.apply_preset(
    {"target_name": [".*attn.*"]},
    # 训练注意力和MLP模块
    {"target_name": [".*mlp.*"]},
    # 训练所有线性层
    {"target_name": [".*linear.*"]}
)

class test_model(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super(test_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

your_model = test_model()

lycoris_net = create_lycoris(
    your_model, 
    1.0, 
    linear_dim=16, 
    linear_alpha=2.0, 
    algo="lora"
)
lycoris_net.apply_to()

# after apply_to(), your_model() will run with LyCORIS net
lycoris_param = lycoris_net.parameters()
print(lycoris_param)
print(your_model)
print(lycoris_net)
# forward_with_lyco = your_model(x)
optimizer_params = lycoris_net.prepare_optimizer_params(lr=1e-4)
print(optimizer_params)
# 检查哪些参数需要梯度
your_model.requires_grad_(False)

trainable_params = sum(p.numel() for p in your_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in your_model.parameters())
print(f"可训练参数: {trainable_params:,}")
print(f"总参数: {total_params:,}")
print(f"可训练比例: {trainable_params/total_params:.2%}")