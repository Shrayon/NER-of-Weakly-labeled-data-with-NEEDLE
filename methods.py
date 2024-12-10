import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns


def visualize_confusion_matrix(true_predictions, true_labels, label_list, save_path):
    flat_true_labels = [label for sublist in true_labels for label in sublist]
    flat_true_predictions = [label for sublist in true_predictions for label in sublist]
    conf_matrix = confusion_matrix(flat_true_labels, flat_true_predictions, labels=label_list)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_list, yticklabels=label_list,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def get_metrics(true_predictions, true_labels, label_list):
    class_metrics = precision_recall_fscore_support(
        y_true=np.concatenate(true_labels),
        y_pred=np.concatenate(true_predictions),
        labels=label_list,
        average=None,
        zero_division=1
    )

    weighted_metrics = precision_recall_fscore_support(
        y_true=np.concatenate(true_labels),
        y_pred=np.concatenate(true_predictions),
        labels=label_list,
        average='weighted',
        zero_division=1
    )
    accuracy = accuracy_score(np.concatenate(true_labels), np.concatenate(true_predictions))
    precision_dict = dict(zip(label_list, class_metrics[0]))
    recall_dict = dict(zip(label_list, class_metrics[1]))
    f1_score_dict = dict(zip(label_list, class_metrics[2]))
    support_dict = dict(zip(label_list, class_metrics[3]))
    precision_df = pd.DataFrame(list(precision_dict.items()), columns=['Label', 'Precision'])
    recall_df = pd.DataFrame(list(recall_dict.items()), columns=['Label', 'Recall'])
    f1_score_df = pd.DataFrame(list(f1_score_dict.items()), columns=['Label', 'F1-Score'])
    support_df = pd.DataFrame(list(support_dict.items()), columns=['Label', 'Support'])
    result_df = precision_df.merge(recall_df, on='Label').merge(f1_score_df, on='Label').merge(support_df, on='Label')
    accuracy_df = pd.DataFrame([{
        'Label': 'accuracy',
        'Precision': accuracy,
        'Recall': np.mean(class_metrics[1]),
        'F1-Score': np.mean(class_metrics[2]),
        'Support': np.sum(class_metrics[3])
    }])
    macro_avg_df = pd.DataFrame([{
        'Label': 'macro avg',
        'Precision': np.mean(class_metrics[0]),
        'Recall': np.mean(class_metrics[1]),
        'F1-Score': np.mean(class_metrics[2]),
        'Support': np.sum(class_metrics[3])
    }])
    weighted_avg_df = pd.DataFrame([{
        'Label': 'weighted avg',
        'Precision': weighted_metrics[0],
        'Recall': weighted_metrics[1],
        'F1-Score': weighted_metrics[2],
        'Support': np.sum(class_metrics[3])
    }])
    result_df = pd.concat([result_df, accuracy_df, macro_avg_df, weighted_avg_df], ignore_index=True)
    result_df = result_df.round(2)
    return result_df
