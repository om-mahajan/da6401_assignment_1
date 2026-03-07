import argparse, json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_data, get_class_names

def parse_arguments():
    p = argparse.ArgumentParser(description='Run inference on test set')
    p.add_argument('--model_path', type=str, default='src/best_model.npy')
    p.add_argument('--config_path', type=str, default='src/best_config.json')
    p.add_argument('-d', '--dataset', type=str, default=None)
    p.add_argument('-b', '--batch_size', type=int, default=256)
    p.add_argument('--log_wandb', action='store_true', default=True,
                   help='Log results to Weights & Biases')
    p.add_argument('-wp', '--wandb_project', type=str, default='da6401_assignment1_mnist')
    p.add_argument('-we', '--wandb_entity', type=str, default='IITmRL')
    return p.parse_args()

def load_model(model_path, config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    model = NeuralNetwork(
        input_size=784, output_size=10, num_layers=cfg['num_layers'],
        hidden_size=cfg['hidden_size'], activation=cfg['activation'],
        weight_init=cfg.get('weight_init', 'xavier'), loss=cfg['loss'],
        optimizer=cfg['optimizer'], lr=cfg['learning_rate'],
        weight_decay=cfg.get('weight_decay', 0.0))
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)
    return model, cfg

def evaluate_model(model, X_test, y_test):
    logits = model.predict(X_test)
    loss, acc = model.evaluate(X_test, y_test)
    y_pred = np.argmax(logits, axis=1)
    
    # Robust true label parsing for Scikit-Learn metrics
    if y_test.ndim == 2 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1) # It's one-hot encoded
    else:
        y_true = y_test.flatten().astype(int) # It's a 1D array of integers
        
    return {
        "logits": logits, "loss": loss, "accuracy": acc,
        "f1": f1_score(y_true, y_pred, average='macro'),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
def main():
    args = parse_arguments()
    model, cfg = load_model(args.model_path, args.config_path)
    dataset = args.dataset or cfg.get('dataset', 'mnist')
    _, _, _, _, X_test, y_test = load_data(dataset)
    results = evaluate_model(model, X_test, y_test)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"Loss:      {results['loss']:.4f}")
    print("Confusion Matrix:")
    print(results['confusion_matrix'])

    if args.log_wandb:
        import wandb
        init_kwargs = {'project': args.wandb_project, 'name': 'inference_results', 'job_type': 'inference'}
        if args.wandb_entity:
            init_kwargs['entity'] = args.wandb_entity
        wandb.init(**init_kwargs)
        wandb.config.update(cfg)
        wandb.log({
            "test_accuracy": results['accuracy'],
            "test_precision": results['precision'],
            "test_recall": results['recall'],
            "test_f1": results['f1'],
            "test_loss": results['loss'],
        })
        class_names = get_class_names(dataset)
        y_pred = np.argmax(results['logits'], axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = results['confusion_matrix']
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title('Confusion Matrix - Inference')
        plt.tight_layout()
        plt.savefig('inference_cm.png', dpi=150)
        plt.close()
        wandb.log({"confusion_matrix": wandb.Image('inference_cm.png')})
        wandb.finish()

    return results

if __name__ == '__main__':
    main()
