import argparse, os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_data, load_raw_images, get_class_names

PROJECT = 'da6401_assignment_1'
ENTITY = None

def wb_init(**kwargs):
    if ENTITY:
        kwargs['entity'] = ENTITY
    kwargs['project'] = get_project()
    return wandb.init(**kwargs)

def get_project():
    return PROJECT

# ============================================================
# Q2.1 - Data Exploration: log 5 samples per class
# ============================================================
def data_exploration(dataset='fashion_mnist'):
    run = wb_init(name=f'data_exploration_{dataset}', job_type='exploration')
    X, y = load_raw_images(dataset)
    class_names = get_class_names(dataset)
    table = wandb.Table(columns=["Image", "Label", "Class Name"])
    for c in range(10):
        idxs = np.where(y == c)[0][:5]
        for i in idxs:
            img = wandb.Image(X[i])
            table.add_data(img, int(c), class_names[c])
    wandb.log({"data_samples": table})
    wandb.finish()

# ============================================================
# Q2.2 - Hyperparameter Sweep
# ============================================================
def sweep_train():
    run = wb_init()
    cfg = dict(wandb.config)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg.get('dataset', 'fashion_mnist'))
    model = NeuralNetwork(
        input_size=784, output_size=10, num_layers=cfg['num_layers'],
        hidden_size=cfg['hidden_size'], activation=cfg['activation'],
        weight_init=cfg.get('weight_init', 'xavier'), loss=cfg['loss'],
        optimizer=cfg['optimizer'], lr=cfg['learning_rate'],
        weight_decay=cfg.get('weight_decay', 0.0))
    for epoch in range(cfg.get('epochs', 10)):
        train_loss, train_acc = model.train_epoch(X_train, y_train, cfg.get('batch_size', 64))
        val_loss, val_acc = model.evaluate(X_val, y_val)
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_accuracy": train_acc,
                    "val_loss": val_loss, "val_accuracy": val_acc})
    test_loss, test_acc = model.evaluate(X_test, y_test)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    test_f1 = f1_score(y_true, y_pred, average='macro')
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1})
    wandb.finish()

def hyperparameter_sweep(count=100):
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'dataset': {'value': 'fashion_mnist'},
            'epochs': {'values': [5, 10]},
            'num_layers': {'values': [2, 3, 4, 5]},
            'hidden_size': {'values': [32, 64, 128]},
            'activation': {'values': ['relu', 'sigmoid', 'tanh']},
            'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'batch_size': {'values': [32, 64]},
            'weight_init': {'values': ['random', 'xavier']},
            'weight_decay': {'values': [0, 0.0005, 0.005]},
            'loss': {'values': ['cross_entropy', 'mean_squared_error']}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=get_project(), entity=ENTITY)
    wandb.agent(sweep_id, function=sweep_train, count=count)

# ============================================================
# Q2.3 - Optimizer Comparison
# ============================================================
def optimizer_comparison(dataset='fashion_mnist'):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset)
    for opt in ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']:
        run = wb_init(name=f'optim_{opt}', group='optimizer_comparison', reinit=True)
        wandb.config.update({'optimizer': opt, 'num_layers': 3, 'hidden_size': 128,
                             'activation': 'relu', 'loss': 'cross_entropy', 'learning_rate': 0.001})
        model = NeuralNetwork(input_size=784, output_size=10, num_layers=3, hidden_size=128,
                              activation='relu', weight_init='xavier', loss='cross_entropy',
                              optimizer=opt, lr=0.001, weight_decay=0.0)
        for epoch in range(10):
            train_loss, train_acc = model.train_epoch(X_train, y_train, 64)
            val_loss, val_acc = model.evaluate(X_val, y_val)
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        test_loss, test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
        wandb.finish()

# ============================================================
# Q2.4 - Vanishing Gradient Analysis
# ============================================================
def vanishing_gradient(dataset='fashion_mnist'):
    X_train, y_train, X_val, y_val, _, _ = load_data(dataset)
    for act in ['sigmoid', 'relu']:
        for nhl in [3, 5]:
            run = wb_init(name=f'vanish_{act}_layers{nhl}',
                             group='vanishing_gradient', reinit=True)
            wandb.config.update({'activation': act, 'num_layers': nhl, 'optimizer': 'adam'})
            model = NeuralNetwork(input_size=784, output_size=10, num_layers=nhl, hidden_size=128,
                                  activation=act, weight_init='xavier', loss='cross_entropy',
                                  optimizer='adam', lr=0.001)
            for epoch in range(10):
                train_loss, train_acc = model.train_epoch(X_train, y_train, 64)
                val_loss, val_acc = model.evaluate(X_val, y_val)
                grad_norms = model.get_gradient_norms()
                log_d = {"epoch": epoch+1, "train_loss": train_loss, "val_accuracy": val_acc}
                for gi, gn in enumerate(grad_norms):
                    log_d[f"grad_norm_layer_{gi}"] = gn
                wandb.log(log_d)
            wandb.finish()

# ============================================================
# Q2.5 - Dead Neuron Investigation
# ============================================================
def dead_neuron(dataset='fashion_mnist'):
    X_train, y_train, X_val, y_val, _, _ = load_data(dataset)
    configs = [('relu', 0.1), ('tanh', 0.1), ('relu', 0.001)]
    for act, lr_val in configs:
        run = wb_init(name=f'dead_{act}_lr{lr_val}',
                         group='dead_neuron', reinit=True)
        wandb.config.update({'activation': act, 'learning_rate': lr_val})
        model = NeuralNetwork(input_size=784, output_size=10, num_layers=3, hidden_size=128,
                              activation=act, weight_init='xavier', loss='cross_entropy',
                              optimizer='adam', lr=lr_val)
        for epoch in range(15):
            train_loss, train_acc = model.train_epoch(X_train, y_train, 64)
            val_loss, val_acc = model.evaluate(X_val, y_val)
            stats = model.get_activation_stats()
            log_d = {"epoch": epoch+1, "train_loss": train_loss, "val_accuracy": val_acc}
            for si, s in enumerate(stats):
                log_d[f"activation_zero_frac_layer_{si}"] = s['zero_fraction']
                log_d[f"activation_mean_layer_{si}"] = s['mean']
                if len(s['values']) > 0:
                    log_d[f"activation_hist_layer_{si}"] = wandb.Histogram(s['values'][:10000])
            wandb.log(log_d)
        wandb.finish()

# ============================================================
# Q2.6 - Loss Function Comparison
# ============================================================
def loss_comparison(dataset='fashion_mnist'):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset)
    for loss_fn in ['cross_entropy', 'mean_squared_error']:
        run = wb_init(name=f'loss_{loss_fn}',
                         group='loss_comparison', reinit=True)
        wandb.config.update({'loss': loss_fn})
        model = NeuralNetwork(input_size=784, output_size=10, num_layers=3, hidden_size=128,
                              activation='relu', weight_init='xavier', loss=loss_fn,
                              optimizer='adam', lr=0.001)
        for epoch in range(10):
            train_loss, train_acc = model.train_epoch(X_train, y_train, 64)
            val_loss, val_acc = model.evaluate(X_val, y_val)
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        test_loss, test_acc = model.evaluate(X_test, y_test)
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
        wandb.finish()

# ============================================================
# Q2.7 - Global Performance Analysis
# ============================================================
def global_analysis():
    api = wandb.Api()
    path = f"{ENTITY}/{get_project()}" if ENTITY else get_project()
    runs = api.runs(path)
    train_accs, test_accs, names = [], [], []
    for r in runs:
        ta = r.summary.get('train_accuracy')
        te = r.summary.get('test_accuracy')
        if ta is not None and te is not None:
            train_accs.append(ta)
            test_accs.append(te)
            names.append(r.name)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(train_accs))
    ax.bar(x, train_accs, alpha=0.6, label='Train Accuracy', width=0.4)
    ax.bar([i+0.4 for i in x], test_accs, alpha=0.6, label='Test Accuracy', width=0.4)
    ax.set_xlabel('Run'); ax.set_ylabel('Accuracy'); ax.legend()
    ax.set_title('Train vs Test Accuracy Across All Runs')
    plt.tight_layout()
    plt.savefig('global_analysis.png', dpi=150)
    run = wb_init(name='global_analysis', job_type='analysis')
    wandb.log({"global_train_vs_test": wandb.Image('global_analysis.png')})
    gap = [(t - e, n) for t, e, n in zip(train_accs, test_accs, names)]
    gap.sort(reverse=True)
    table = wandb.Table(columns=["Run", "Train Acc", "Test Acc", "Gap"])
    for g, n in gap[:10]:
        idx = names.index(n)
        table.add_data(n, train_accs[idx], test_accs[idx], g)
    wandb.log({"overfitting_runs": table})
    wandb.finish()

# ============================================================
# Q2.8 - Confusion Matrix for Best Model
# ============================================================
def confusion_matrix_best(dataset='fashion_mnist', model_path='models/best_model.npy', config_path='models/best_config.json'):
    with open(config_path) as f:
        cfg = json.load(f)
    model = NeuralNetwork(input_size=784, output_size=10, num_layers=cfg['num_layers'],
                          hidden_size=cfg['hidden_size'], activation=cfg['activation'],
                          weight_init=cfg.get('weight_init', 'xavier'), loss=cfg['loss'],
                          optimizer=cfg['optimizer'], lr=cfg['learning_rate'])
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)
    _, _, _, _, X_test, y_test = load_data(dataset)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_names = get_class_names(dataset)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix - Best Model')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    run = wb_init(name='confusion_matrix', job_type='analysis')
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})
    misclassified = np.where(y_pred != y_true)[0]
    table = wandb.Table(columns=["Image", "True Label", "Predicted Label"])
    for idx in misclassified[:50]:
        img = (X_test[idx].reshape(28, 28) * 255).astype(np.uint8)
        table.add_data(wandb.Image(img), class_names[y_true[idx]], class_names[y_pred[idx]])
    wandb.log({"misclassified_samples": table})
    wandb.finish()

# ============================================================
# Q2.9 - Weight Initialization & Symmetry
# ============================================================
def weight_init_symmetry(dataset='fashion_mnist'):
    X_train, y_train, X_val, y_val, _, _ = load_data(dataset)
    for init_method in ['zeros', 'xavier']:
        run = wb_init(name=f'init_{init_method}',
                         group='weight_init_symmetry', reinit=True)
        wandb.config.update({'weight_init': init_method})
        model = NeuralNetwork(input_size=784, output_size=10, num_layers=3, hidden_size=128,
                              activation='relu', weight_init=init_method, loss='cross_entropy',
                              optimizer='adam', lr=0.001)
        indices = np.random.permutation(X_train.shape[0])
        iteration = 0
        for _ in range(5):
            for start in range(0, X_train.shape[0], 64):
                if iteration >= 50:
                    break
                idx = indices[start:start+64]
                xb, yb = X_train[idx], y_train[idx]
                y_pred = model.forward(xb)
                model.backward(yb, y_pred)
                log_d = {"iteration": iteration}
                first_hidden_grad = model.layers[0].grad_W
                for neuron_idx in range(min(5, first_hidden_grad.shape[1])):
                    log_d[f"grad_neuron_{neuron_idx}"] = np.linalg.norm(first_hidden_grad[:, neuron_idx])
                log_d["grad_norm_layer_0"] = np.linalg.norm(first_hidden_grad)
                wandb.log(log_d)
                model.update_weights()
                iteration += 1
            if iteration >= 50:
                break
        wandb.finish()

# ============================================================
# Q2.10 - Fashion-MNIST Transfer Challenge
# ============================================================
def fashion_transfer():
    configs = [
        {"num_layers": 3, "hidden_size": 128, "activation": "relu", "optimizer": "adam", "lr": 0.001},
        {"num_layers": 4, "hidden_size": 128, "activation": "relu", "optimizer": "nadam", "lr": 0.001},
        {"num_layers": 3, "hidden_size": 64, "activation": "tanh", "optimizer": "adam", "lr": 0.001},
    ]
    X_train, y_train, X_val, y_val, X_test, y_test = load_data('fashion_mnist')
    for ci, cfg in enumerate(configs):
        run = wb_init(name=f'fashion_transfer_config{ci+1}',
                         group='fashion_transfer', reinit=True)
        wandb.config.update(cfg)
        model = NeuralNetwork(input_size=784, output_size=10, num_layers=cfg['num_layers'],
                              hidden_size=cfg['hidden_size'], activation=cfg['activation'],
                              weight_init='xavier', loss='cross_entropy',
                              optimizer=cfg['optimizer'], lr=cfg['lr'])
        for epoch in range(10):
            train_loss, train_acc = model.train_epoch(X_train, y_train, 64)
            val_loss, val_acc = model.evaluate(X_val, y_val)
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        test_loss, test_acc = model.evaluate(X_test, y_test)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        test_f1 = f1_score(y_true, y_pred, average='macro')
        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1})
        print(f"Config {ci+1}: test_acc={test_acc:.4f}, test_f1={test_f1:.4f}")
        if ci == 0:
            os.makedirs('models', exist_ok=True)
            np.save('models/best_model.npy', model.get_weights())
            with open('models/best_config.json', 'w') as f:
                json.dump({**cfg, 'dataset': 'fashion_mnist', 'loss': 'cross_entropy',
                          'weight_init': 'xavier', 'learning_rate': cfg['lr'],
                          'weight_decay': 0.0, 'epochs': 10, 'batch_size': 64}, f, indent=2)
        wandb.finish()

# ============================================================
# Main CLI dispatcher
# ============================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=str, required=True,
                   choices=['data_exploration', 'sweep', 'optimizer_comparison',
                            'vanishing_gradient', 'dead_neuron', 'loss_comparison',
                            'global_analysis', 'confusion_matrix', 'weight_init_symmetry',
                            'fashion_transfer'])
    p.add_argument('-wp', '--wandb_project', type=str, default='da6401_assignment_1')
    p.add_argument('--sweep_count', type=int, default=100)
    p.add_argument('--dataset', type=str, default='fashion_mnist')
    p.add_argument('-we', '--wandb_entity', type=str, default=None)
    args = p.parse_args()
    PROJECT = args.wandb_project
    ENTITY = args.wandb_entity

    experiments = {
        'data_exploration': lambda: data_exploration(args.dataset),
        'sweep': lambda: hyperparameter_sweep(args.sweep_count),
        'optimizer_comparison': lambda: optimizer_comparison(args.dataset),
        'vanishing_gradient': lambda: vanishing_gradient(args.dataset),
        'dead_neuron': lambda: dead_neuron(args.dataset),
        'loss_comparison': lambda: loss_comparison(args.dataset),
        'global_analysis': global_analysis,
        'confusion_matrix': lambda: confusion_matrix_best(args.dataset),
        'weight_init_symmetry': lambda: weight_init_symmetry(args.dataset),
        'fashion_transfer': fashion_transfer,
    }
    experiments[args.experiment]()
