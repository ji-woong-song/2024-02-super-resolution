import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms


class SRCNN(nn.Module):
    def __init__(self, num_channels=3, base_channels=64, num_blocks=1):
        """
        Args:
            num_channels (int): 입력 이미지의 채널 수 (RGB=3, Grayscale=1)
            base_channels (int): 기본 피처맵 개수
            num_blocks (int): 중간 컨볼루션 블록의 개수
        """
        super(SRCNN, self).__init__()
        
        # 첫 번째 레이어: 패치 추출과 표현
        self.first_layer = nn.Sequential(
            nn.Conv2d(num_channels, base_channels, kernel_size=9, padding=4),
            nn.ReLU(inplace=True)
        )
        
        # 중간 레이어: 비선형 매핑
        middle_layers = []
        for _ in range(num_blocks):
            middle_layers.extend([
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.middle_layers = nn.Sequential(*middle_layers)
        
        # 마지막 레이어: 복원
        self.last_layer = nn.Conv2d(base_channels, num_channels, kernel_size=5, padding=2)
        
        # 가중치 초기화
        self._initialize_weights()
        
    def forward(self, x):
        out = self.first_layer(x)
        out = self.middle_layers(out)
        out = self.last_layer(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ... existing SRCNN class code ...

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale_factor=3):
        """
        Args:
            lr_dir (str): 저해상도 이미지가 있는 디렉토리 경로
            hr_dir (str): 고해상도 이미지가 있는 디렉토리 경로
            scale_factor (int): 업스케일링 비율 (3배)
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.image_files = os.listdir(lr_dir)

        self.lr_transform = transforms.Compose([
            transforms.Resize((570, 564), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        
        self.hr_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 이미지 로드
        lr_img = Image.open(os.path.join(self.lr_dir, img_name)).convert('RGB')
        hr_img = Image.open(os.path.join(self.hr_dir, img_name)).convert('RGB')
        
        # 저해상도 이미지를 3배 업스케일링
        lr_upscaled = self.lr_transform(lr_img)
        hr_tensor = self.hr_transform(hr_img)

        return lr_upscaled, hr_tensor

# DataLoader 설정
def get_dataloader(lr_dir, hr_dir, batch_size=32, num_workers=4):
    dataset = SRDataset(lr_dir, hr_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

# 사용 예시
lr_folder = "lr"
hr_folder = "hr"
train_dataloader = get_dataloader(lr_folder, hr_folder)


def train_srcnn(model, train_dataloader, num_epochs, device='cuda', 
                learning_rate=0.001, save_path='srcnn_model.pth'):
    """
    SRCNN 모델을 학습시키고 loss를 기록하는 함수
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # loss 기록을 위한 변수들
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # tqdm으로 진행률 표시
        with tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for lr_images, hr_images in pbar:
                # 데이터를 디바이스로 이동
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파
                outputs = model(lr_images)
                
                # 손실 계산
                loss = criterion(outputs, hr_images)
                
                # 역전파 및 옵티마이저 스텝
                loss.backward()
                optimizer.step()
                
                # 현재 loss 기록
                current_loss = loss.item()
                epoch_losses.append(current_loss)
                
                # 진행바에 현재 loss 표시
                pbar.set_postfix({'loss': f'{current_loss:.6f}'})
        
        # 에폭의 평균 손실 계산 및 기록
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f'Model saved at epoch {epoch+1} with loss {best_loss:.6f}')
    
    return loss_history

# 학습 실행 예시
if __name__ == "__main__":
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 모델 및 데이터로더 초기화
    model = SRCNN()
    lr_folder = "lr"
    hr_folder = "hr"
    train_dataloader = get_dataloader(lr_folder, hr_folder, batch_size=25)
    
    # 학습 실행
    num_epochs = 50
    loss_history = train_srcnn(
        model=model,
        train_dataloader=train_dataloader,
        num_epochs=num_epochs,
        device=device,
        learning_rate=0.001,
        save_path='srcnn_best_model.pth'
    )

    # Loss 그래프 시각화 (선택사항)
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()