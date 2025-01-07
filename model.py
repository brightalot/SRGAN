import math
import torch
from torch import nn

# Generator Network
class Generator(nn.Module):
    def __init__(self, scale_factor):
        # Upsample Block의 반복 횟수 계산 (scale_factor에 따라 결정)
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        
        # [k9n64s1] - Generator의 초기 Conv 블록
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        # 16개의 Residual Block 추가
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
        )

        # [k3n64s1] - Skip Connection 이후 Conv + BatchNorm
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # [k3n256s1 -> PixelShuffle x2] - Upsample Blocks
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        
        # [k9n3s1] - 최종 Conv 블록 (RGB 이미지 출력)
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        # 초기 Conv 블록
        block1 = self.block1(x)
        
        # 16개의 Residual Block
        residual_out = self.residual_blocks(block1)  

        # Skip Connection
        block7 = self.block7(residual_out)
        
        # Upsample Blocks 및 Skip Connection 통합
        block8 = self.block8(block1 + block7)

        # Tanh 정규화하여 출력 (RGB 이미지 범위로 변환)
        return (torch.tanh(block8) + 1) / 2


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # [k3n64s1] - 초기 Conv 블록
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            # [k3n64s2] - 다운샘플링 Conv 블록
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # [k3n128s1] - 채널 수 증가
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # [k3n128s2] - 다운샘플링 Conv 블록
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # [k3n256s1] - 채널 수 증가
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # [k3n256s2] - 다운샘플링 Conv 블록
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # [k3n512s1] - 채널 수 증가
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # [k3n512s2] - 다운샘플링 Conv 블록
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Dense Layers
            nn.AdaptiveAvgPool2d(1),  # Adaptive Pooling으로 고정 크기 텐서 생성
            nn.Conv2d(512, 1024, kernel_size=1),  # Dense(1024)
            nn.LeakyReLU(0.2),  # Leaky ReLU 활성화
            nn.Conv2d(1024, 1, kernel_size=1),  # Dense(1)
            nn.Sigmoid()  # Sigmoid 활성화
        )


    def forward(self, x):
        # 배치 크기 계산 및 Sigmoid 출력
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


# Residual Block (Generator 내부)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # [k3n64s1] - 첫 번째 Conv + BatchNorm + PReLU
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        # [k3n64s1] - 두 번째 Conv + BatchNorm
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 두 개의 Conv-BN 연산 및 Skip Connection
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


# Upsample Block (Generator 내부)
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        # [k3n256s1] - Conv 레이어 (PixelShuffle 전)
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        # PixelShuffle (해상도 증가)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # PReLU 활성화 함수
        self.prelu = nn.PReLU()

    def forward(self, x):
        # Conv → PixelShuffle → PReLU
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
