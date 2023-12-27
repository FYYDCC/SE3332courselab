import os
import cv2 as cv
import torch
from torch import nn
import timm
from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loguru import logger

# Define the ClassificationModel class (if not already defined)
class ClassificationModel(nn.Module):
    def __init__(self, num_classes, hidden_size, num_lstm_layers=2, backbone_name='resnet101'):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.adap = nn.AdaptiveAvgPool2d((2, 2))
        self.lstm = nn.LSTM(2048, hidden_size, num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch, num_frames, channels, height, width = x.shape
        x = torch.reshape(x, (-1, *x.shape[2:]))
        x1, x2, x3, x4, x5 = self.backbone(x)
        x = self.adap(x3)
        x = nn.Flatten()(x)
        x = torch.reshape(x, (batch, num_frames, -1))
        x, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, ...]
        x = self.fc(x)
        return x

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_classes=10, num_frames=20, transform=None, target_transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.num_frames = num_frames

        self.video_filename_list = []
        self.classesIdx_list = []

        self.class_dict = {class_label: idx for idx, class_label in enumerate(
            sorted(os.listdir(self.data_dir)))}

        for class_label, class_idx in self.class_dict.items():
            class_dir = os.path.join(self.data_dir, class_label)
            for video_filename in sorted(os.listdir(class_dir)):
                self.video_filename_list.append(
                    os.path.join(class_label, video_filename))
                self.classesIdx_list.append(class_idx)

    def __len__(self):
        return len(self.video_filename_list)

    def read_video(video_path, transform=None):
        frames = []
        cap = cv.VideoCapture(video_path)
        count_frames = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if transform:
                    transformed = transform(image=frame)
                    frame = transformed['image']

                frames.append(frame)
                count_frames += 1
            else:
                break
        stride = count_frames // 20
        new_frames = []
        count = 0
        for i in range(0, count_frames, stride):
            if count >= 20:
                break
            new_frames.append(frames[i])
            count += 1

        cap.release()

        return torch.stack(new_frames, dim=0)

    def __getitem__(self, idx):
        classIdx = self.classesIdx_list[idx]
        video_filename = self.video_filename_list[idx]
        video_path = os.path.join(self.data_dir, video_filename)
        frames = self.read_video(video_path)
        return frames, classIdx

# Function to load the pre-trained model
def load_pretrained_model(model, pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
    model.eval()

# Function to make predictions on a single video
def predict_single_video(model, video_path, transform, device='cpu'):
    frames = VideoDataset.read_video(video_path, transform)
    frames = frames.unsqueeze(0)
    frames = frames.to(device)

    with torch.no_grad():
        output = model(frames)

    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

if __name__ == '__main__':
    num_classes = 10
    hidden_size = 256
    img_size = (120, 160)

    transform = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    model = ClassificationModel(num_classes=num_classes, hidden_size=hidden_size)
    pretrained_model_path = "Resnet101pretrained_model.pth"
    load_pretrained_model(model, pretrained_model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    video_path = "your_path" #change to your file path
    predicted_class = predict_single_video(model, video_path, transform, device=device)

    print(f"Predicted Class Index:{predicted_class}")
