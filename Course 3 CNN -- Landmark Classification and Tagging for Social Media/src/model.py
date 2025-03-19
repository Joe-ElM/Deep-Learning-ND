import torch
import torch.nn as nn
import torch
import torch.nn as nn


# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 3x224x224 -> 16x224x224
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2),  # -> 16x112x112
            
            nn.Conv2d(16, 32, 3, padding=1),  # -> 32x112x112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2),  # -> 32x56x56

            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2),  # -> 64x28x28

            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x28x28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2),  # -> 128x14x14

            nn.Conv2d(128, 256, 3, padding=1),  # -> 256x14x14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2),  # -> 256x7x7

            nn.Conv2d(256, 512, 3, padding=1),  # -> 512x7x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2, 2),  # -> 512x3x3

            nn.Flatten(),  # 512 * 3 * 3 = 4608

            nn.Linear(512 * 3 * 3, 512),  # -> 512
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(512, num_classes),  # Final output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    
# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.4) -> None:
#         super().__init__()

#         # Feature Extraction
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3x244x244 -> 32x244x244
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64x244x244
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(2, 2),  # -> 64x122x122

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> 128x122x122
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),  # -> 256x122x122
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(2, 2),  # -> 256x61x61

#             nn.Conv2d(256, 512, kernel_size=3, padding=1),  # -> 512x61x61
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(2, 2),  # -> 512x30x30
#         )

#         # Adaptive Average Pooling Instead of Flattening
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # -> 512x1x1

#         # Fully Connected Classifier
#         self.classifier = nn.Sequential(
#             nn.Flatten(),  # -> 512
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(512, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.model(x)            # Feature extractor
#         x = self.global_avg_pool(x)  # Adaptive pooling
#         x = self.classifier(x)       # Fully connected classifier
#         return x
    
    
# class MyModel(nn.Module):
#     def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

#         super().__init__()

#         # YOUR CODE HERE
#         # Define a CNN architecture. Remember to use the variable num_classes
#         # to size appropriately the output of your classifier, and if you use
#         # the Dropout layer, use the variable "dropout" to indicate how much
#         # to use (like nn.Dropout(p=dropout))

#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # 3x224x224 -> 16x224x224
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # -> 16x112x112

#             nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> 32x112x112
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # -> 32x56x56

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64x56x56
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # -> 64x28x28

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> 128x28x28
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # -> 128x14x14
#         )

#         # YOUR CODE HERE
#         # Compute the feature size manually
#         # If the input image size is 224x224, the final feature map size is 128x14x14
#         flattened_size = 128 * 14 * 14  # = 25088

#         # Fully Connected Classifier
#         self.classifier = nn.Sequential(
#             nn.Flatten(),  # to 25088
#             nn.Linear(flattened_size, 256),  # -> 256
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # YOUR CODE HERE: process the input tensor through the
#         # feature extractor, the pooling and the final linear
#         # layers (if appropriate for the architecture chosen)      
        
#         x = self.model(x)  # Pass input through CNN feature extractor
#         x = self.classifier(x)  # Pass flattened output through classifier
#         return x





######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
