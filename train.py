import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import classification_report


class PropertyClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PropertyClassifierNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ARTIFACTS_DIR = "dataset_artifacts"
    MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "property_classifier_model.pth")

    if not os.path.exists(ARTIFACTS_DIR):
        logging.error(
            f"Помилка: Папку '{
                ARTIFACTS_DIR
            }' не знайдено. Спочатку запустіть prepare_data.py"
        )
        return

    logging.info("Завантаження даних...")
    X_train = torch.load(os.path.join(ARTIFACTS_DIR, "X_train.pt"))
    y_train = torch.load(os.path.join(ARTIFACTS_DIR, "y_train.pt"))
    X_test = torch.load(os.path.join(ARTIFACTS_DIR, "X_test.pt"))
    y_test = torch.load(os.path.join(ARTIFACTS_DIR, "y_test.pt"))

    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    model = PropertyClassifierNN(input_dim, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 20
    logging.info(f"Починаємо тренування на {epochs} епох...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / y_test.size(0)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_loader)
            logging.info(
                f"Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {
                    accuracy * 100:.1f}%"
            )

    logging.info("Генерація детальної статистики по категоріях...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, y_pred = torch.max(test_outputs, 1)

        y_true_numpy = y_test.cpu().numpy()
        y_pred_numpy = y_pred.cpu().numpy()

        target_names = label_encoder.classes_

        report = classification_report(
            y_true_numpy, y_pred_numpy, target_names=target_names
        )
        print("\nЗвіт по категоріях:")
        print(report)

    logging.info("Збереження моделі...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Ваги моделі успішно збережено у: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
