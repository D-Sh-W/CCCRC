from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, fbeta_score, precision_score, \
    precision_recall_curve, auc

from DataSet import *
from FusedModel import *
from utils import *


def test(model, dataloader, device, pred_thre):
    all_labels, all_preds, all_probs = [], [], []
    wrong_samples = []

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as t:
            t.set_description(f"Test")
            for dm_images, freq_images, labels, paths in t:
                dm_images = dm_images.to(device)
                freq_images = freq_images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(dm_images, freq_images)
                probs = torch.sigmoid(outputs).detach().cpu().numpy()

                preds = probs > pred_thre
                labels_np = labels.cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels_np)

                for i in range(len(paths)):
                    pred_label = int(preds[i][0])
                    true_label = int(labels_np[i][0])
                    if pred_label != true_label:
                        wrong_samples.append([paths[i], pred_label, true_label])

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    f2 = fbeta_score(all_labels, all_preds, beta=2)

    cm = confusion_matrix(all_labels, all_preds)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0

    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)

    return acc, recall, precision, f1, f2, fpr, cm, wrong_samples, precision_curve, recall_curve, pr_auc


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data_type = "test_1_1"
    # test_data_type = "test_1_10"
    pred_thre = 0.7
    test_csv = f"./{test_data_type}.csv"
    model_path = "model/best_model.pth"
    test_root_dir = f"./{test_data_type}/prediction_result/"
    os.makedirs(test_root_dir, exist_ok=True)
    confusion_matrix_dir = test_root_dir + "test_cm"
    pr_auc_dir = test_root_dir + "test_pr_auc"
    test_metrics_path = test_root_dir + "test_metrics.csv"
    wrong_samples_path = test_root_dir + "wrong_predictions.csv"

    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = CustomDataset(test_csv, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FusedModel()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"GPUs: {torch.cuda.device_count()}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()

    acc, recall, precision, f1, f2, fpr, cm, wrong_samples, precision_curve, recall_curve, pr_auc = test(model,
                                                                                                         test_loader,
                                                                                                         device,
                                                                                                         pred_thre)

    save_test_confusion_matrix(cm, "Test", confusion_matrix_dir)
    save_test_PR_AUC(recall_curve, precision_curve, pr_auc, pr_auc_dir)

    with open(test_metrics_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Accuracy", "Recall", "Precision", "F1", "F2", "FPR", "PR-AUC"])
        writer.writerow([f"{acc:.4f}", f"{recall:.4f}", f"{precision:.4f}", f"{f1:.4f}", f"{f2:.4f}", f"{fpr:.4f}",
                         f"{pr_auc:.4f}"])

    with open(wrong_samples_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image_Path", "Predicted_Label", "True_Label"])
        writer.writerows(wrong_samples)

    print(
        f"Test Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, F2: {f2:.4f}, FPR: {fpr:.4f}, PR-AUC: {pr_auc:.4f}")
