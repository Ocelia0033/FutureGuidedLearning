# train_pv.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from models.FutureGuidedLearner import FutureGuidedLearner

# ===============================
# 1. 读取光伏数据
# ===============================
data = pd.read_csv('data/pv_data.csv')

features = ["Global_Horizontal_Radiation", "Pyranometer_1", "Wind_Speed", "Temperature_Probe_1"]
target = ["Active_Power"]

data = data.dropna().reset_index(drop=True)

# 数据标准化
scaler = MinMaxScaler()
data[features + target] = scaler.fit_transform(data[features + target])

# ===============================
# 2. 自定义数据集
# ===============================
class PVDataset(Dataset):
    def __init__(self, df, input_len=24, pred_len=6):
        self.df = df
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.df) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.df.iloc[idx:idx+self.input_len][features].values
        y = self.df.iloc[idx+self.input_len:idx+self.input_len+self.pred_len][target].values
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

dataset = PVDataset(data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ===============================
# 3. 定义模型
# ===============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FutureGuidedLearner(
    input_dim=len(features),
    output_dim=len(target),
    input_len=24,
    pred_len=6
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# ===============================
# 4. 训练循环
# ===============================
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(loader):.6f}")

torch.save(model.state_dict(), "pv_model.pth")
print("✅ 模型训练完成，权重已保存为 pv_model.pth")
import matplotlib.pyplot as plt

model.eval()
x, y = dataset[1000]
with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device)).cpu().numpy().flatten()
real = y.numpy().flatten()

plt.figure(figsize=(8,4))
plt.plot(real, label='真实值')
plt.plot(pred, label='预测值')
plt.legend()
plt.title("光伏功率预测结果")
plt.show()
