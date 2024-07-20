import os
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 데이터 준비
# 임의의 데이터 생성, 실제 사용 시 적절한 데이터로 대체
import torch.utils.data as data

class CustomDataset(data.Dataset):
    def __init__(self, backbone_folder, encoder_folder):
        self.backbone_folder = backbone_folder
        self.encoder_folder = encoder_folder

    def __getitem__(self, index):
        backbone_file = f"{self.backbone_folder}/backbone output/{index}.pt"
        encoder_file = f"{self.encoder_folder}/encoder output/{index}.pt"
        backbone_output = torch.load(backbone_file)
        encoder_output = torch.load(encoder_file)
        return backbone_output, encoder_output

    def __len__(self):
        # Assuming the number of files in the folders are the same
        return len(os.listdir(self.backbone_folder))

backbone_folder = "dataset/hiddenvec/backbone_output"
encoder_folder = "dataset/hiddenvec/encoder_output"
dataset = CustomDataset(backbone_folder, encoder_folder)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 모델 정의
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(10, 1)  # 10차원 입력, 1차원 출력

    def forward(self, x):
        return self.linear(x)

model = RegressionModel()

# 3. 학습 과정
criterion = nn.CrossEntropyLoss()  # 크로스 엔트로피 손실
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 옵티마이저

# 학습 루프
for epoch in range(100):  # 예: 100 에폭
    for backbone_output, encoder_output in dataloader:
        optimizer.zero_grad()   # 그래디언트 초기화
        outputs = model(backbone_output)  # 모델 예측
        loss = criterion(outputs, encoder_output)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 매개변수 업데이트

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

