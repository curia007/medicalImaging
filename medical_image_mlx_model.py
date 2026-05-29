import os
import argparse
import numpy as np
from PIL import Image

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


IMAGE_SIZE = 128


def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[None, :, :]  # channel first
    return arr


def load_dataset(data_dir):
    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    label_to_id = {name: i for i, name in enumerate(labels)}

    images = []
    targets = []

    for label in labels:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(load_image(os.path.join(folder, file)))
                targets.append(label_to_id[label])

    X = mx.array(np.stack(images))
    y = mx.array(np.array(targets, dtype=np.int32))

    return X, y, labels


class MedicalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        flattened = 64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8)
        self.fc1 = nn.Linear(flattened, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def __call__(self, x):
        x = self.pool(nn.relu(self.conv1(x)))
        x = self.pool(nn.relu(self.conv2(x)))
        x = self.pool(nn.relu(self.conv3(x)))

        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


def loss_fn(model, X, y):
    logits = model(X)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def accuracy(model, X, y):
    logits = model(X)
    preds = mx.argmax(logits, axis=1)
    return mx.mean(preds == y)


def train(data_dir, epochs=20, lr=0.001):
    X, y, labels = load_dataset(data_dir)

    num_classes = len(labels)
    model = MedicalCNN(num_classes)

    optimizer = optim.Adam(learning_rate=lr)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        loss, grads = loss_and_grad(model, X, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        acc = accuracy(model, X, y)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {float(loss):.4f} | "
            f"Accuracy: {float(acc):.4f}"
        )

    mx.savez("medical_model_weights.npz", **dict(model.parameters()))
    np.save("medical_labels.npy", np.array(labels))

    print("Saved model: medical_model_weights.npz")
    print("Saved labels: medical_labels.npy")


def predict(image_path):
    labels = np.load("medical_labels.npy", allow_pickle=True).tolist()

    model = MedicalCNN(len(labels))
    weights = mx.load("medical_model_weights.npz")
    model.update(weights)

    img = mx.array(load_image(image_path)[None, :, :, :])
    logits = model(img)
    probs = nn.softmax(logits, axis=1)

    pred_id = int(mx.argmax(probs, axis=1)[0])
    confidence = float(probs[0, pred_id])

    print(f"Prediction: {labels[pred_id]}")
    print(f"Confidence: {confidence:.2%}")
    print("Warning: research use only, not a medical diagnosis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--data-dir", help="Folder containing labeled image folders")
    parser.add_argument("--image", help="Image path for prediction")
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir, args.epochs)

    if args.mode == "predict":
        predict(args.image)