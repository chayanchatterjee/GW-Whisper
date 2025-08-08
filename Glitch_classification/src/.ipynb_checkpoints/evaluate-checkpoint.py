import argparse
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import WhisperModel
from dataset import one_channel_LigoBinaryData
from model import one_channel_ligo_binary_classifier
from peft import LoraConfig, get_peft_model
import fnmatch


def load_best_model(encoder, device, lora_weights_path, dense_weights_path, num_classes, method, lora_rank, lora_alpha):
    whisper_model = WhisperModel.from_pretrained(f"openai/whisper-{encoder}").encoder.to(device)

    module_names = [name for name, module in whisper_model.named_modules()]
    patterns = ["layers.*.self_attn.q_proj", "layers.*.self_attn.k_proj", "layers.*.self_attn.v_proj", "layers.*.self_attn.o_proj"]
    matched_modules = []
    for pattern in patterns:
        matched_modules.extend(fnmatch.filter(module_names, pattern))

    if method == 'DoRA':
        lora_config = LoraConfig(use_dora=True, r=lora_rank, lora_alpha=lora_alpha, target_modules=matched_modules)
    else:
        lora_config = LoraConfig(use_dora=False, r=lora_rank, lora_alpha=lora_alpha, target_modules=matched_modules)

    whisper_model_with_lora = get_peft_model(whisper_model, lora_config).to(device)
    model = one_channel_ligo_binary_classifier(whisper_model_with_lora, num_classes=num_classes).to(device)

    model.encoder.load_state_dict(torch.load(lora_weights_path, map_location=device))
    model.classifier.load_state_dict(torch.load(dense_weights_path, map_location=device))

    return model

def evaluate_test_set(model, test_loader, device, label_encoder, results_path, model_name):
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, snr in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(label_encoder.classes_)))

    # Save classification report
    report_path = os.path.join(results_path, f"{model_name}_test_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Plot and save confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    cm_path = os.path.join(results_path, f"{model_name}_test_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    test_ds = load_from_disk(args.test_data_path)

    label_encoder = LabelEncoder()
    all_labels = test_ds['labels']

    modified_labels = ["GW" if label == "GW" else " ".join(label.split("_")).title() for label in all_labels]
    label_encoder.fit(modified_labels)

    test_ds = test_ds.map(lambda x: {'encoded_labels': label_encoder.transform(["GW" if x['labels'] == "GW" else " ".join(x['labels'].split("_")).title()])[0]}, batched=False)

    test_data = one_channel_LigoBinaryData(test_ds, device, encoder=args.encoder)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load best model
    model = load_best_model(
        encoder=args.encoder,
        device=device,
        lora_weights_path=args.lora_weights_path,
        dense_weights_path=args.dense_weights_path,
        num_classes=len(label_encoder.classes_),
        method=args.method,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

    print("Evaluating model on test dataset...")
    evaluate_test_set(model, test_loader, device, label_encoder, args.results_path, args.model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--results_path", type=str, default="Glitch_classification/results/generic", help="Path to save results")
    parser.add_argument("--encoder", type=str, default="tiny", help="Whisper encoder size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--model_name", type=str, default="multi_class_model", help="Name of the model")
    parser.add_argument("--method", type=str, choices=['LoRA', 'DoRA'], required=True, help="Method to apply (LoRA or DoRA)")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="Path to the best LoRA weights file")
    parser.add_argument("--dense_weights_path", type=str, required=True, help="Path to the best dense weights file")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha for LoRA")

    args = parser.parse_args()
    main(args)
