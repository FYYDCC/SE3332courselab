from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import torch
import torchvision
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from tqdm import tqdm
import random
from loguru import logger
import matplotlib.pyplot as plt

# 自定义的SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积层
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 修改全连接层
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # 注意修改这里的输入大小
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # 输入大小：(batch_size, num_frames, 3, 64, 64)
        batch_size, num_frames, _, _ = x.shape
        x = x.view(-1, 3, 64, 64)  # 展平 num_frames
        x = self.pool1(self.relu1(self.conv1(x)))
        # 输出大小：(batch_size * num_frames, 64, 32, 32)

        x = self.pool2(self.relu2(self.conv2(x)))
        # 输出大小：(batch_size * num_frames, 128, 16, 16)

        x = self.pool3(self.relu3(self.conv3(x)))
        # 输出大小：(batch_size * num_frames, 256, 8, 8)

        x = x.view(batch_size, num_frames, -1)  # 恢复 num_frames
        x = x.view(-1, 256 * 8 * 8)  # 展平
        x = self.relu4(self.fc1(x))
        # 输出大小：(batch_size * num_frames, 512)

        x = x.view(batch_size, num_frames, -1)    # 恢复 num_frames
        x = x.view(-1, 512, 1, 1)  # 添加额外的维度 (batch_size * num_frames, 512, 1, 1)

        return x

# 修改上述代码使用SimpleCNN
class VideoDataset(torch.utils.data.Dataset):
    '''
    Custom Dataset for loading videos and their class labels
    '''
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

    def read_video(self, video_path):
        frames = []
        cap = cv.VideoCapture(video_path)
        count_frames = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if self.transform:
                    transformed = self.transform(image=frame)
                    frame = transformed['image']

                frames.append(frame)
                count_frames += 1
            else:
                break

        stride = count_frames // self.num_frames
        new_frames = []
        count = 0
        for i in range(0, count_frames, stride):
            if count >= self.num_frames:
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


class ClassificationModel(nn.Module):
    def __init__(self, num_classes, hidden_size, num_lstm_layers=2, backbone_name='simple_cnn'):
        super().__init__()

        if backbone_name == 'simple_cnn':
            self.backbone = SimpleCNN(num_classes=num_classes)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        
        self.adap = nn.AdaptiveAvgPool2d((2, 2))

        self.lstm = nn.LSTM(9600, hidden_size, num_lstm_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        'x: batch, num_frames, channels, height, width'
        batch, num_frames, channels, height, width = x.shape

        # x: batch * num_frames, channels, height, width 
        x = torch.reshape(x, (-1, *x.shape[2:]))
        x = self.backbone(x)

        # x: batch * num_frames, 512, 2, 2
        x = self.adap(x)

        # x: batch * num_frames, 512, 2, 2
        x = x.view(batch * num_frames, -1)

        # x: batch, num_frames, features vector 
        x = torch.reshape(x, (batch, num_frames, -1))

        # x: Tensor kích thước (batch_size, sequence_length, hidden_size)
        # h_n: Hidden state cuối cùng của lớp cuối cùng, kích thước (num_layers, batch_size, hidden_size)
        # c_n: Cell state cuối cùng của lớp cuối cùng, kích thước (num_layers, batch_size, hidden_size)
        x, (h_n, c_n) = self.lstm(x)

        x = h_n[-1, ...]

        x = self.fc(x)

        return x
    
    
def train(model, train_data, loss_fn, optimizer, epochs, weights=None, save_last_weights_path="cnn_model.pth",
          save_best_weights_path=None, freeze=False, steps_per_epoch=None,
          device='cpu', validation_data=None, validation_split=None, scheduler=None):

    # set device
    if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif isinstance(device, torch.device):
        device = device
    else:
        device = torch.device('cpu')

    # chia dữ liệu thành 2 tập train và val    
    if validation_data is not None:
        val_data = validation_data
    elif validation_split is not None:
        train_data, val_data = train_test_split(train_data, test_size=validation_split, random_state=42)
    else:
        val_data = None

    # save best model
    if save_best_weights_path:
        if val_data is None:
            train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        best_loss, _ = evaluate(model, val_data, device=device, loss_fn=loss_fn)

    # đặt số lần update weights trong 1 epoch
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)

    num_steps = len(train_data)
    count_steps = 1

    ## History
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }

    # add model to device
    model = model.to(device)

    ############################### Train and Val ##########################################
    for epoch in range(1, epochs + 1):
        # tạo mới iterator ở đầu mỗi epoch
        iterator = iter(train_data)

        # tính tổng giá trị hàm mất mát cho mỗi epoch
        running_loss = 0.
        train_correct = 0
        train_total = steps_per_epoch * train_data.batch_size

        # đặt model ở chế độ huấn luyện 
        model.train()

        for step in tqdm(range(steps_per_epoch), desc=f'epoch: {epoch}/{epochs}: ', ncols=80):
            try:
                img_batch, label_batch = next(iterator)
            except StopIteration:
                # nếu iterator hết dữ liệu, tạo mới iterator
                iterator = iter(train_data)
                img_batch, label_batch = next(iterator)

            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            # Xóa các gradient
            optimizer.zero_grad()

            # tính toán đầu ra
            output_batch = model(img_batch)

            # tính loss
            loss = loss_fn(output_batch, label_batch.long())

            # lan truyền ngược
            loss.backward()

            # cập nhật trọng số
            optimizer.step()

            # tính giá trị trung bình loss qua mỗi epoch
            running_loss += loss.item()

            # tính toán accuracy
            _, predicted = torch.max(output_batch.data, 1)
            train_correct += (predicted == label_batch).sum().item()

            count_steps += 1

        # tính giá trị loss và accuracy trung bình qua mỗi epoch
        train_loss = running_loss / steps_per_epoch
        train_acc = train_correct / train_total

        # validate model nếu có tập validation
        if val_data is not None:
            val_loss, val_acc = evaluate(model, val_data, device=device, loss_fn=loss_fn)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            logger.info(f'Epoch {epoch}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        else:
            logger.info(f'Epoch {epoch}/{epochs}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')

        # lưu trọng số của epoch hiện tại
        if save_last_weights_path:
            torch.save(model.state_dict(), save_last_weights_path)

        # lưu trọng số tốt nhất
        if save_best_weights_path:
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_best_weights_path)

        # điều chỉnh learning rate theo scheduler nếu được cung cấp
        if scheduler:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

    logger.info('Finished Training')
    return history



def evaluate(model, val_data, device='cpu', loss_fn=None):
    # chuyển model sang chế độ đánh giá
    model.eval()

    val_correct = 0
    val_total = len(val_data.dataset)

    running_loss = 0.0

    # không tính gradient trong quá trình đánh giá
    with torch.no_grad():
        for img_batch, label_batch in tqdm(val_data, desc='Evaluating', ncols=100):
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)

            # tính toán đầu ra
            output_batch = model(img_batch)

            # tính loss nếu có
            if loss_fn:
                loss = loss_fn(output_batch, label_batch.long())
                running_loss += loss.item()

            # tính toán accuracy
            _, predicted = torch.max(output_batch.data, 1)
            val_correct += (predicted == label_batch).sum().item()

    # tính giá trị trung bình loss qua tất cả các batch
    val_loss = running_loss / len(val_data) if loss_fn else None

    # tính giá trị accuracy
    val_acc = val_correct / val_total

    logger.info(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

    return val_loss, val_acc


def visualize_history(history):
    # Visualize training history
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'])
    if 'val_acc' in history:
        plt.plot(history['val_acc'])
        plt.legend(['Train', 'Validation'])
    else:
        plt.legend(['Train'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    if 'val_loss' in history:
        plt.plot(history['val_loss'])
        plt.legend(['Train', 'Validation'])
    else:
        plt.legend(['Train'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('cnn_8e-6.png')
    plt.show()


# 使用SimpleCNN的训练过程
if __name__ == '__main__':
    num_classes = 10
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_frames = 20
    img_size = (120, 160)
    num_workers = 8
    hidden_size = 256  # You can adjust this value based on your preference

    # Define transformations and dataset
    transform = A.Compose(
        [
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    logger.info('Loading dataset')
    full_dataset = VideoDataset(data_dir="lab3_data/data", num_frames=num_frames, num_classes=num_classes, transform=transform)

    # Split dataset into train and test
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logger.info('Dataset loaded')

    # Create an instance of the model
    model = ClassificationModel(num_classes=num_classes, hidden_size=hidden_size, backbone_name='simple_cnn')

    # Define loss function, optimizer, and scheduler if needed
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000008)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)

    # Train the model
    history = train(model, train_loader, loss_fn, optimizer, epochs=50, device=device,
                    validation_data=test_loader, scheduler=scheduler)

    # Visualize the training history
    visualize_history(history)
