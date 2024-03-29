import sys
from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support
from .nc_dataset_class import *


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def load_and_save(name, data_dir=None, classname=None):
    if not data_dir:
        data_dir = os.path.abspath("./data/")
    if os.path.exists(os.path.join(data_dir, name)):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(data_dir, name, name + '_graph.bin')
        # Load the Homogeneous Graph
        g = load_graphs(graph_path)[0][0]
        info_path = os.path.join(data_dir, name, name + '_info.pkl')
        n_classes = load_info(info_path)['num_classes']
        print("NumNodes: ", g.num_nodes())
        print("NumEdges: ", g.num_edges())
        print("NumFeats: ", g.ndata["feat"].shape[1])
        print("NumClasses: ", n_classes)
        print("NumTrainingSamples: ", torch.count_nonzero(g.ndata["train_mask"]).item())
        print("NumValidationSamples: ", torch.count_nonzero(g.ndata["val_mask"]).item())
        print("NumTestSamples: ", torch.count_nonzero(g.ndata["test_mask"]).item())
        return g, n_classes
    else:
        if classname:
            dataset = str_to_class(classname)(save_path=data_dir)
        else:
            dataset = str_to_class(name)(save_path=data_dir)

        dataset.save_data()
        # Load the Homogeneous Graph as an UndirectedGraph
        n_classes = dataset.num_classes
        return dataset[0], n_classes


def compute_metrics(y_pred, y_true):
    with torch.no_grad():
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        metrics_out = dict()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        metrics_out["auc"] = auc(fpr, tpr)
        metrics_out["precision"], metrics_out["recall"], metrics_out["fscore"], _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=1   )
        return metrics_out


def min_max_scaler(x):
    x_std = (x - x.min(axis=0)[0]) / (x.max(axis=0)[0] - x.min(axis=0)[0])
    x_scaled = x_std * (x.max - x.min) + x.min
    return x_scaled


def get_sample_weights(labels):
    class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    sample_weights = torch.tensor([weight[t] for t in labels])
    return sample_weights
