
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        # Squeeze: Global average pooling
        b, c, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)
        
        # Excitation: FC-ReLU-FC-Sigmoid
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)
        
        # Scale: Channel-wise multiplication
        return x * excitation.expand_as(x)


class PreActResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 dropout_rate=0.1, use_se=True):
        super(PreActResidualBlock, self).__init__()
        
        # Pre-activation untuk conv1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        
        # Pre-activation untuk conv2
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        
        # Squeeze-and-Excitation
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels, reduction=16)
        
        # Dropout untuk regularisasi
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        # Downsample untuk skip connection
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # Pre-activation Block 1: BN-GELU-Conv
        out = self.bn1(x)
        out = F.gelu(out)
        out = self.conv1(out)
        
        # Pre-activation Block 2: BN-GELU-Conv
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.conv2(out)
        
        # SE attention
        if self.use_se:
            out = self.se(out)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        
        return out


class ResNet34Improved(nn.Module):
    """
    ResNet-34 dengan modifikasi:
    - SE Block untuk channel attention
    - Pre-activation untuk stabilitas
    - GELU activation
    - Dropout untuk regularisasi
    """
    def __init__(self, num_classes=5, dropout_rate=0.2, use_se=True):
        super(ResNet34Improved, self).__init__()
        
        # Stem: Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet stages dengan Pre-activation blocks
        self.stage1 = self._make_stage(64, 64, 3, stride=1, dropout_rate=dropout_rate, use_se=use_se)
        self.stage2 = self._make_stage(64, 128, 4, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.stage3 = self._make_stage(128, 256, 6, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.stage4 = self._make_stage(256, 512, 3, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        
        # Final batch norm (untuk pre-activation)
        self.bn_final = nn.BatchNorm2d(512)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)
        
        # Weight initialization
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, dropout_rate, use_se):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        # First block may have stride > 1 and different channels
        layers.append(PreActResidualBlock(
            in_channels, out_channels, stride, downsample, 
            dropout_rate=dropout_rate, use_se=use_se
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(PreActResidualBlock(
                out_channels, out_channels, stride=1, 
                dropout_rate=dropout_rate, use_se=use_se
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Kaiming initialization untuk Conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.maxpool(x)
        
        # Stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Final activation
        x = self.bn_final(x)
        x = F.gelu(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc(x)
        x = self.fc(x)
        
        return x


def create_resnet34_improved(num_classes=5, dropout_rate=0.2, use_se=True):
    """
    Factory function untuk membuat ResNet-34 Improved
    
    Args:
        num_classes: Jumlah kelas output
        dropout_rate: Dropout rate untuk regularisasi (0.0-0.5)
        use_se: Aktifkan SE block atau tidak
    """
    return ResNet34Improved(num_classes=num_classes, dropout_rate=dropout_rate, use_se=use_se)


def test_model():
    """Test model architecture"""
    print("Creating ResNet-34 Improved Model...")
    print("Modifications:")
    print("  1. Squeeze-and-Excitation (SE) Blocks")
    print("  2. Pre-activation Residual Blocks")
    print("  3. GELU Activation")
    print("  4. Dropout Regularization")
    print("="*60)
    
    model = create_resnet34_improved(num_classes=5, dropout_rate=0.2, use_se=True)
    
    try:
        summary(model, input_size=(1, 3, 224, 224), verbose=1)
    except Exception as e:
        print(f"Error in summary: {e}")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            output = model(test_input)
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    model = test_model()
    print("\n" + "="*60)
    print("MODEL READY FOR TRAINING!")
    print("="*60)
