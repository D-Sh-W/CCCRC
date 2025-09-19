from time import sleep
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.parallel import DataParallel

from FusedModel import *
from DataSet import *
from utils import *


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer):
    model.train()
    all_labels, all_preds = [], []
    total_loss = 0

    with tqdm(dataloader, unit="batch") as t:
        for dm_images, freq_images, labels, paths in t:
            t.set_description(f"Train")
            dm_images = dm_images.to(device)
            freq_images = freq_images.to(device)

            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(dm_images, freq_images)

            loss = criterion(outputs, labels, epoch)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0

    writer.add_scalar("Train/Loss", total_loss / len(dataloader), epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    writer.add_scalar("Train/Recall", recall, epoch)
    writer.add_scalar("Train/FPR", fpr, epoch)

    return acc, recall, fpr, total_loss / len(dataloader), cm


def validate(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as v:
            v.set_description(f"Val  ")
            for dm_images, freq_images, labels, paths in v:
                dm_images = dm_images.to(device)
                freq_images = freq_images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(dm_images, freq_images)
                loss = criterion(outputs, labels, epoch)
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds = probs > 0.5

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0

    writer.add_scalar("Val/Loss", total_loss / len(dataloader), epoch)
    writer.add_scalar("Val/Accuracy", acc, epoch)
    writer.add_scalar("Val/Recall", recall, epoch)
    writer.add_scalar("Val/FPR", fpr, epoch)

    return acc, recall, fpr, total_loss / len(dataloader), cm


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_csv = "./train.csv"
    val_csv = "./val.csv"

    result_path = "./result/result.csv"
    best_model_path = "model/best_model.pth"
    best_epoch_info_path = "./best_epoch/best_epoch_info.txt"
    log_dir = "./log"
    confusion_matrix_dir = "./confusion_matrix"

    batch_size = 32
    epoches = 100
    best_epoch = 0
    early_stopper = EarlyStopping(patience=10, delta=0.0001)

    summary_writer = SummaryWriter(log_dir=log_dir)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(train_csv, transform)
    val_dataset = CustomDataset(val_csv, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FusedModel()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"GPUs: {torch.cuda.device_count()}")
    model.to(device)

    criterion = DynamicWeightedBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    with open(result_path, "a", newline="", encoding="utf-8") as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(
            ["train_acc", "train_rec", "train_fpr", "train_loss", "val_acc", "val_rec", "val_fpr", "val_loss"])

    for epoch in range(100):
        print(f"Epoch {epoch + 1} / {epoches}")

        sleep(0.01)

        train_acc, train_rec, train_fpr, train_loss, train_cm = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, summary_writer
        )

        sleep(0.01)

        print(
            f"Train Acc: {train_acc:.4f}, Recall: {train_rec:.4f}, FPR: {train_fpr:.4f}, Loss: {train_loss:.4f}")

        sleep(0.01)

        val_acc, val_rec, val_fpr, val_loss, val_cm = validate(
            model, val_loader, criterion, device, epoch, summary_writer)

        sleep(0.01)

        print(
            f"Val Acc: {val_acc:.4f}, Recall: {val_rec:.4f}, FPR: {val_fpr:.4f}, Loss: {val_loss:.4f}")

        sleep(0.01)

        with open(result_path, "a", newline="", encoding="utf-8") as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(
                [f"{train_acc:.4f}", f"{train_rec:.4f}", f"{train_fpr:.4f}", f"{train_loss:.4f}", f"{val_acc:.4f}",
                 f"{val_rec:.4f}", f"{val_fpr:.4f}", f"{val_loss:.4f}"])

        save_confusion_matrix(train_cm, epoch, "Train", confusion_matrix_dir)
        save_confusion_matrix(val_cm, epoch, "Val", confusion_matrix_dir)

        scheduler.step(val_acc)

        early_stopper.step(val_loss)

        if early_stopper.should_save:
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch + 1
            train_info = [train_acc, train_rec, train_fpr, train_loss]
            val_info = [val_acc, val_rec, val_fpr, val_loss]
            save_best_epoch_info(best_epoch_info_path, epoch, train_info, val_info)
            print("Best model (based on val_loss) saved.")
        if early_stopper.should_stop:
            print(f"Early stopping triggered!")
            break

        print(f"Best epoch: {best_epoch}")

        sleep(0.01)

    summary_writer.close()


if __name__ == '__main__':
    main()
