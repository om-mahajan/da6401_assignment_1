import numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

OPENML_IDS = {"mnist": "mnist_784", "fashion_mnist": "Fashion-MNIST"}

def _fetch(dataset):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mlp_data")
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, f"{dataset}.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return (data['X_train'], data['y_train']), (data['X_test'], data['y_test'])
    print(f"Fetching {dataset} from OpenML (one-time download)...")
    d = fetch_openml(OPENML_IDS[dataset], version=1, as_frame=False, parser='liac-arff')
    X, y = d.data.astype(np.uint8), d.target.astype(np.int64)
    X = X.reshape(-1, 28, 28)
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    np.savez_compressed(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    return (X_train, y_train), (X_test, y_test)

def load_data(dataset="fashion_mnist", val_split=0.1):
    (X_train, y_train), (X_test, y_test) = _fetch(dataset)
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0
    num_classes = 10
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        X_train, y_train_oh, test_size=val_split, random_state=42, stratify=y_train)
    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh

def load_raw_images(dataset="fashion_mnist"):
    (X, y), _ = _fetch(dataset)
    return X, y

CLASS_NAMES_FASHION = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CLASS_NAMES_MNIST = [str(i) for i in range(10)]

def get_class_names(dataset):
    return CLASS_NAMES_FASHION if dataset == "fashion_mnist" else CLASS_NAMES_MNIST
