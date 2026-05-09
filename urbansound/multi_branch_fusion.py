import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNNBranch(nn.Module):
    def __init__(self, out_features=256, freeze_pretrained=False):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # The original pretrained resnet takes 3-channel (RGB) images.
        # Mel spectrograms are 1-channel. We must replace the very first Conv2d layer
        # while keeping the same output channels, kernel size, and stride.
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias,
        )

        if freeze_pretrained:
            # Freeze all parameters of the ResNet backbone
            for param in resnet.parameters():
                param.requires_grad = False
            # Unfreeze the new conv1 layer since its weights are randomly initialized
            for param in resnet.conv1.parameters():
                param.requires_grad = True

        # Dynamically get the feature dimension of the backbone (e.g. 512 for ResNet18, 2048 for ResNet50)
        in_features = resnet.fc.in_features

        # Remove the final fully connected classification layer (fc) from ResNet.
        # We only want it as a feature extractor.
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Map the backbone features to our desired out_features
        self.fc = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    def forward(self, x):
        # Input shape expected: (Batch, 1, Mel_Bands, Time_Frames)
        x = self.feature_extractor(x)
        # ResNet output x is now shape (Batch, in_features, 1, 1).
        # Flatten it to just (Batch, in_features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GRUBranch(nn.Module):
    def __init__(self, input_features, hidden_size=128):
        super().__init__()
        # Input shape: (Batch, Time_Steps, Features)
        self.gru = nn.GRU(input_features, hidden_size, batch_first=True)

    def forward(self, x):
        _, h_n = self.gru(x)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # return the final hidden state of the top layer
        return h_n[-1]


class EngineeredBranch(nn.Module):
    def __init__(self, input_features, out_features=64):
        super().__init__()
        # Input shape: (Batch, Total_Features)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_features),
            nn.Linear(input_features, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)


class MultiBranchAttentionFusion(nn.Module):
    def __init__(
        self,
        num_classes=10,
        cnn_out=256,
        gru_in_features=40,
        gru_out=128,
        eng_in_features=20,
        eng_out=64,
        hidden_dim=128,
    ):
        super().__init__()

        self.cnn_branch = CNNBranch(out_features=cnn_out, freeze_pretrained=True)
        self.gru_branch = GRUBranch(input_features=gru_in_features, hidden_size=gru_out)
        self.eng_branch = EngineeredBranch(
            input_features=eng_in_features, out_features=eng_out
        )

        # Projections to shared latent space D
        self.proj_c = nn.Linear(cnn_out, hidden_dim)
        self.proj_g = nn.Linear(gru_out, hidden_dim)
        self.proj_e = nn.Linear(eng_out, hidden_dim)

        # Attention network
        # Takes the hidden representations and produces a score for each
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            # Use CrossEntropyLoss during training, which includes Softmax
        )

    def forward(self, x_cnn, x_gru, x_eng):
        # Extract features from each branch
        feat_c = self.cnn_branch(x_cnn)
        feat_g = self.gru_branch(x_gru)
        feat_e = self.eng_branch(x_eng)

        # Project to uniform hidden dimension D
        h_c = F.relu(self.proj_c(feat_c))
        h_g = F.relu(self.proj_g(feat_g))
        h_e = F.relu(self.proj_e(feat_e))

        # Stack into H: (Batch, 3, D)
        H = torch.stack((h_c, h_g, h_e), dim=1)

        # Calculate attention scores
        # Output shape of attention(H): (Batch, 3, 1)
        att_scores = self.attention(H)

        # Alpha: Softmax over the 3 branches (dim=1)
        alpha = F.softmax(att_scores, dim=1)

        # Weighted Fusion
        # Multiply H by alpha and sum across branches (dim=1)
        # (Batch, 3, D) * (Batch, 3, 1) -> (Batch, 3, D) -> sum -> (Batch, D)
        f = torch.sum(alpha * H, dim=1)

        # Classification
        out = self.classifier(f)
        return out
