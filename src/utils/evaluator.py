from tqdm import tqdm
import numpy as np
import torch
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

logger = logging.getLogger(__name__)


class Evaluator:
    def end_of_epoch_eval(self, all_predictions, all_labels, prefix: str):
        # Calculate accuracy
        try:
            if all_labels.shape[1] > 1:  # Multi-class case (one-hot encoded)
                # Get predicted classes (argmax of predictions)
                predicted_classes = np.argmax(all_predictions, axis=1)
                # Get true classes (argmax of one-hot labels)
                true_classes = np.argmax(all_labels, axis=1)
                # Calculate accuracy
                accuracy = accuracy_score(true_classes, predicted_classes)
            else:  # Binary case
                # For binary case, convert predictions to binary (0 or 1)
                predicted_classes = (all_predictions.flatten() > 0.5).astype(int)
                true_classes = all_labels.flatten().astype(int)
                accuracy = accuracy_score(true_classes, predicted_classes)
        except Exception as e:
            logger.warning(f"Could not calculate accuracy: {e}")
            accuracy = 0.0

        # Convert one-hot labels to binary for each class
        # For multi-class ROC AUC, we need to use one-vs-rest approach
        try:
            if all_labels.shape[1] > 1:  # Multi-class case (one-hot encoded)
                # Calculate ROC AUC for each class (one-vs-rest)
                roc_auc_scores = []
                pr_auc_scores = []

                for class_idx in range(all_labels.shape[1]):
                    class_labels = all_labels[:, class_idx]
                    class_predictions = (
                        all_predictions[:, class_idx]
                        if all_predictions.shape[1] > 1
                        else all_predictions.flatten()
                    )

                    # Only calculate metrics if class has both positive and negative samples
                    if len(np.unique(class_labels)) > 1:
                        roc_auc = roc_auc_score(class_labels, class_predictions)
                        pr_auc = average_precision_score(
                            class_labels, class_predictions
                        )
                        roc_auc_scores.append(roc_auc)
                        pr_auc_scores.append(pr_auc)

                # Calculate macro-averaged metrics
                roc_auc = np.mean(roc_auc_scores) if roc_auc_scores else 0.0
                pr_auc = np.mean(pr_auc_scores) if pr_auc_scores else 0.0
            else:  # Binary case
                roc_auc = roc_auc_score(all_labels.flatten(), all_predictions.flatten())
                pr_auc = average_precision_score(
                    all_labels.flatten(), all_predictions.flatten()
                )
        except Exception as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
            roc_auc = 0.0
            pr_auc = 0.0

        acc_k = self.acc_at_k(all_predictions, all_labels, K=1)
        acc_k3 = self.acc_at_k(all_predictions, all_labels, K=3)
        acc_k5 = self.acc_at_k(all_predictions, all_labels, K=5)

        return {
            f"{prefix}/accuracy": accuracy,
            f"{prefix}/roc_auc": roc_auc,
            f"{prefix}/pr_auc": pr_auc,
            f"{prefix}/acc_k": acc_k,
            f"{prefix}/acc_k3": acc_k3,
            f"{prefix}/acc_k5": acc_k5,
        }

    def acc_at_k(
        self, all_predictions: np.ndarray, all_labels: np.ndarray, K: int
    ) -> float:
        """
        Compute the flexible Acc@K metric as defined in the paper.

        Acc@K treats predictions within ±floor(K/2) period-class distance
        from the true class as correct.

        Args:
            all_predictions: Model outputs of shape (N, C) for multi-class logits/scores
                             or (N,) / (N, 1) for binary outputs.
            all_labels: One-hot labels of shape (N, C) for multi-class, or
                        (N,) / (N, 1) for binary labels.
            K: Window size parameter. Allowed distance is floor(K/2).

        Returns:
            Acc@K value in [0, 1].
        """
        if all_predictions is None or all_labels is None:
            return 0.0
        try:
            # Normalize shapes
            preds = np.asarray(all_predictions)
            labels = np.asarray(all_labels)

            # Determine class indices
            if labels.ndim == 2 and labels.shape[1] > 1:
                true_idx = np.argmax(labels, axis=1)
            else:
                # Binary labels: convert to {0,1}
                true_idx = labels.reshape(-1)
                # If labels are not already {0,1}, threshold at 0.5
                if true_idx.dtype != np.int64 and true_idx.dtype != np.int32:
                    true_idx = (true_idx > 0.5).astype(int)

            if preds.ndim == 2 and preds.shape[1] > 1:
                pred_idx = np.argmax(preds, axis=1)
            else:
                # Binary predictions: threshold at 0.5
                pred_idx = (preds.reshape(-1) > 0.5).astype(int)

            if true_idx.shape[0] == 0:
                return 0.0

            allowed_distance = K // 2
            correct = np.abs(pred_idx - true_idx) <= allowed_distance
            acc_at_k = float(np.mean(correct))
            return acc_at_k
        except Exception as e:
            logger.warning(f"Could not calculate Acc@K (K={K}): {e}")
            return 0.0
