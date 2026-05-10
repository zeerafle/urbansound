import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNNBranch(nn.Module):
    def __init__(self, out_features=256, freeze_pretrained=False):
        super().__init__()

        # Load ConvNeXt
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

        # 1. First Layer Fix: Handle 1-channel Mel Spectrograms
        original_conv1 = base.features[0][0]

        # Create the new layer
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        # Preserve and adapt the weights (sum across the 3 RGB channels)
        with torch.no_grad():
            # Original weight shape is [out_channels, 3, kernel_h, kernel_w]
            # We sum across dim 1 to get [out_channels, 1, kernel_h, kernel_w]
            new_conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

            # Preserve the bias if it exists
            if original_conv1.bias is not None:
                new_conv1.bias.data = original_conv1.bias.data.clone()

        base.features[0][0] = new_conv1

        # 2. Transients Fix: Use Max Pooling for sharp sounds like gunshots/horns
        base.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        # 3. Parameter Management: Freeze the pretrained weights
        if freeze_pretrained:
            for param in base.parameters():
                param.requires_grad = False
            # Ensure the new 1-channel conv layer CAN learn
            for param in base.features[0][0].parameters():
                param.requires_grad = True

        # 4. Dimension Handling
        in_features = base.classifier[2].in_features  # 1024 for base

        # Strip the ImageNet classifier head
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        # 5. The Projection Layer (Replacing your 'self.classifier')
        # We use standard LayerNorm here because we will flatten first.
        self.projection = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),  # Normalized in the 1D latent space
            nn.ReLU(),
            nn.Dropout(
                0.3
            ),  # Adding dropout here helps with the UrbanSound8K imbalance
        )

    def forward(self, x):
        # x: (Batch, 1, Mels, Time)
        x = self.feature_extractor(x)  # Output: (Batch, 1024, 1, 1)
        x = torch.flatten(x, 1)  # Output: (Batch, 1024)

        # This is your projection to the Fusion space
        x = self.projection(x)  # Output: (Batch, out_features)
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
        h_c = self.proj_c(feat_c)
        h_g = self.proj_g(feat_g)
        h_e = self.proj_e(feat_e)

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
        return out, alpha
