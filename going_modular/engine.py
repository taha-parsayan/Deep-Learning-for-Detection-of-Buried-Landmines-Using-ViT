import torch
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch."""
    model.train()
    train_loss, correct_preds, total_preds = 0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimizer steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        correct_preds += (torch.argmax(y_pred, dim=1) == y).sum().item()
        total_preds += len(y)

    train_loss = train_loss / len(dataloader)
    train_acc = correct_preds / total_preds
    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, Dict[str, float], str]:
    """Tests a PyTorch model for a single epoch and calculates metrics for binary, 3-class, or 5-class classification."""
    model.eval()
    test_loss, correct_preds, total_preds = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred_logits = model(X)

            # Calculate loss
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()

            # Predictions
            y_pred_labels = torch.argmax(y_pred_logits, dim=1)
            correct_preds += (y_pred_labels == y).sum().item()
            total_preds += len(y)

            # Store predictions for metrics
            all_preds.extend(y_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Accuracy
    test_acc = correct_preds / total_preds
    test_loss = test_loss / len(dataloader)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Sensitivity, Specificity, and F1 per class
    metrics = {}
    for class_index in range(len(cm)):
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        
        # Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f'class_{class_index}_sensitivity'] = sensitivity
        metrics[f'class_{class_index}_specificity'] = specificity

    # Weighted F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Classification report for detailed per-class metrics
    classification_summary = classification_report(all_labels, all_preds)

    # Add the overall F1 score to metrics
    metrics['weighted_f1_score'] = f1

    return test_loss, test_acc, metrics, classification_summary


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model, returning evaluation metrics."""
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_metrics": [],
        "classification_summaries": []
    }

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc, metrics, classification_summary = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Log progress
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        for class_index, sensitivity in metrics.items():
            print(f"{class_index}: {sensitivity:.4f}")

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_metrics"].append(metrics)
        results["classification_summaries"].append(classification_summary)

    return results
