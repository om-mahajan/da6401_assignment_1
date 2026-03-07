import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import wandb
from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_data

def parse_arguments():
    p = argparse.ArgumentParser(description='Train a neural network')
    p.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    p.add_argument('-e', '--epochs', type=int, default=10)
    p.add_argument('-b', '--batch_size', type=int, default=64)
    p.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'])
    p.add_argument('-o', '--optimizer', type=str, default='adam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    p.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    p.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    p.add_argument('-nhl', '--num_layers', type=int, default=3)
    p.add_argument('-sz', '--hidden_size', type=int, default=128)
    p.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    p.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier', 'zeros'])
    p.add_argument('-wp', '--wandb_project', type=str, default='da6401_assignment_1')
    p.add_argument('-we', '--wandb_entity', type=str, default=None)
    p.add_argument('-wn', '--wandb_name', type=str, default=None)
    p.add_argument('--save_path', type=str, default='models/')
    p.add_argument('--sweep', action='store_true')
    args, _ = p.parse_known_args()
    return args

def build_run_name(cfg):
    return f"{cfg['optimizer']}_{cfg['activation']}_hl{cfg['num_layers']}_hs{cfg['hidden_size']}_lr{cfg['learning_rate']}"

def train(config=None, sweep_mode=False):
    args = parse_arguments() if not sweep_mode else argparse.Namespace()
    if sweep_mode:
        run = wandb.init()
        cfg = dict(wandb.config)
    else:
        cfg = {
            'dataset': args.dataset, 'epochs': args.epochs, 'batch_size': args.batch_size,
            'loss': args.loss, 'optimizer': args.optimizer, 'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay, 'num_layers': args.num_layers,
            'hidden_size': args.hidden_size, 'activation': args.activation,
            'weight_init': args.weight_init
        }
        run_name = args.wandb_name or build_run_name(cfg)
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=cfg)
        cfg = dict(wandb.config)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg.get('dataset', 'fashion_mnist'))
    model = NeuralNetwork(
        input_size=784, output_size=10, num_layers=cfg['num_layers'],
        hidden_size=cfg['hidden_size'], activation=cfg['activation'],
        weight_init=cfg.get('weight_init', 'xavier'), loss=cfg['loss'],
        optimizer=cfg['optimizer'], lr=cfg['learning_rate'],
        weight_decay=cfg.get('weight_decay', 0.0))

    best_val_acc = 0
    for epoch in range(cfg.get('epochs', 10)):
        train_loss, train_acc = model.train_epoch(X_train, y_train, cfg.get('batch_size', 64))
        val_loss, val_acc = model.evaluate(X_val, y_val)
        grad_norms = model.get_gradient_norms()
        log_dict = {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc,
                     "val_loss": val_loss, "val_accuracy": val_acc}
        for gi, gn in enumerate(grad_norms):
            log_dict[f"grad_norm_layer_{gi}"] = gn
        wandb.log(log_dict)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

    test_loss, test_acc = model.evaluate(X_test, y_test)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    if not sweep_mode:
        save_dir = getattr(args, 'save_path', 'models/')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'best_model.npy'), best_weights)
        with open(os.path.join(save_dir, 'best_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

    wandb.finish()
    return model

def main():
    args = parse_arguments()
    if args.sweep:
        train(sweep_mode=True)
    else:
        train()

if __name__ == '__main__':
    main()
