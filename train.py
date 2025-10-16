import argparse, os, time, json, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# ---------- Utils ----------

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def plot_curves(history_csv, out_png):
    df = pd.read_csv(history_csv)
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_acc.png"))
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_loss.png"))
    plt.close()

def plot_confmat(cm, classes, out_png):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,6))
    plt.imshow(cm_norm, interpolation='nearest')
    plt.title('Confusion Matrix (normalized)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = f"{cm_norm[i, j]:.2f}"
        plt.text(j, i, txt, horizontalalignment="center",
                 color="white" if cm_norm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------- Data ----------

def get_loaders(batch_size=128):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

# ---------- Model ----------

def get_model(num_classes=10):
    # 轻量：ResNet18 (torchvision) -> 改最后FC
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ---------- Train / Eval ----------

def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
    acc = correct / total
    loss = loss_sum / total
    return acc, loss, torch.cat(all_preds), torch.cat(all_labels)

def train_baseline(args, device):
    run_dir = Path("runs")/ "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir/"config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, classes = get_loaders(args.batch_size)
    model = get_model(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    hist_rows = []
    log_txt = open(run_dir/"train_log_baseline.txt", "w")
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[baseline] epoch {epoch}/{args.epochs}")
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=loss_sum/total, acc=correct/total)
        sched.step()

        train_acc = correct/total
        train_loss = loss_sum/total
        val_acc, val_loss, vpred, vtrue = evaluate(model, val_loader, device)

        log_line = f"epoch={epoch} train_acc={train_acc:.4f} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        print(log_line); print(log_line, file=log_txt, flush=True)

        hist_rows.append({"epoch":epoch,"train_acc":train_acc,"train_loss":train_loss,"val_acc":val_acc,"val_loss":val_loss})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), run_dir/"best.pt")
            cm = confusion_matrix(vtrue.numpy(), vpred.numpy())
            plot_confmat(cm, classes, str(run_dir/"confmat_best.png"))

    log_txt.close()
    pd.DataFrame(hist_rows).to_csv(run_dir/"history.csv", index=False)
    plot_curves(str(run_dir/"history.csv"), str(run_dir/"curves.png"))
    print(f"[baseline] best_acc={best_acc:.4f}")

def kd_loss(student_logits, teacher_logits, T=2.0):
    log_p = F.log_softmax(student_logits/T, dim=1)
    q = F.softmax(teacher_logits/T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T*T)

def train_self_distill(args, device):
    assert args.teacher_ckpt and os.path.exists(args.teacher_ckpt), "teacher_ckpt 路径不存在"
    run_dir = Path("runs")/ "selfdistill"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir/"config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, classes = get_loaders(args.batch_size)

    # teacher / student 同构
    teacher = get_model(num_classes=10).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = get_model(num_classes=10).to(device)

    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    hist_rows = []
    log_txt = open(run_dir/"train_log_selfdistill.txt", "w")
    for epoch in range(1, args.epochs+1):
        student.train()
        pbar = tqdm(train_loader, desc=f"[selfKD] epoch {epoch}/{args.epochs}")
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)

            loss_ce = ce(s_logits, y)
            loss_kd = kd_loss(s_logits, t_logits, T=args.temperature)
            loss = args.alpha*loss_kd + (1-args.alpha)*loss_ce

            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            correct += (s_logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=loss_sum/total, acc=correct/total)
        sched.step()

        train_acc = correct/total
        train_loss = loss_sum/total
        val_acc, val_loss, vpred, vtrue = evaluate(student, val_loader, device)

        log_line = f"epoch={epoch} train_acc={train_acc:.4f} train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        print(log_line); print(log_line, file=log_txt, flush=True)

        hist_rows.append({"epoch":epoch,"train_acc":train_acc,"train_loss":train_loss,"val_acc":val_acc,"val_loss":val_loss})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), run_dir/"best.pt")
            cm = confusion_matrix(vtrue.numpy(), vpred.numpy())
            plot_confmat(cm, classes, str(run_dir/"confmat_best.png"))

    log_txt.close()
    pd.DataFrame(hist_rows).to_csv(run_dir/"history.csv", index=False)
    plot_curves(str(run_dir/"history.csv"), str(run_dir/"curves.png"))
    print(f"[selfKD] best_acc={best_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=["baseline","selfdistill"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.7, help="KD loss 权重")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD温度")
    parser.add_argument("--teacher_ckpt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print("Device:", device)

    if args.exp=="baseline":
        train_baseline(args, device)
    else:
        train_self_distill(args, device)

if __name__ == "__main__":
    main()