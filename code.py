import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from skmultilearn.dataset import load_from_arff
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def load_data(data, target, batch_size, test_ratio):
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, target, test_size=test_ratio, random_state=args.seed)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, output_size=-1, activation='relu', dropout=0.2):
        super(MLP, self).__init__()
        if args.activation == 'relu':
            activation_function = torch.nn.ReLU()
        elif args.activation == 'tanh':
            activation_function = torch.nn.Tanh()
        elif args.activation == 'sigmoid':
            activation_function = torch.nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function {args.activation}")

        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_layers[0]))
        layers.append(activation_function)
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation_function)
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))

        if output_size != -1:
            layers.append(torch.nn.Linear(hidden_layers[-1], output_size))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def run_server(rank, args):
    dist.init_process_group(backend='gloo', init_method=args.init_method, rank=rank, world_size=args.n_client + 1)
    model = MLP(args.hidden_layers_client[-1], args.hidden_layers_server, output_size=2, activation=args.activation, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) if args.optimizer == 'adam' else torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    
    work_counter = 0
    stop_counter = 0
    total_loss = 0
    losses = []
    epochs = []

    while True:
        # The combined1 tensor contains the current batch size and the task type (training or testing)
        combined1 = torch.zeros(2, dtype=torch.long)
        client_id = dist.recv(tensor=combined1)
        batch_size, task_type = combined1.tolist()

        if task_type == -100: # Received stop signal
            stop_counter += 1
            if stop_counter == args.n_client:
                break
            else:
                continue
        
        # The combined2 tensor contains the cut layer and the label
        combined2 = torch.zeros((batch_size, args.hidden_layers_client[-1] + 1), dtype=torch.float32)
        dist.recv(tensor=combined2, src=client_id)
        cut_layer, target = combined2[:, :-1], combined2[:, -1].long()    

        if task_type == 0: # Training
            model.train()
            cut_layer, target = cut_layer.to(args.device), target.to(args.device)
            cut_layer.requires_grad = True
            output = model(cut_layer)
            loss = torch.nn.functional.cross_entropy(output, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dist.send(tensor=cut_layer.grad.detach().cpu(), dst=client_id)

            work_counter += 1
            if work_counter % args.n_report == 0:
                print(f"Server: Processed {work_counter} requests, Average Loss {total_loss / work_counter}")
                losses.append(total_loss / work_counter)
                epochs.append(work_counter)

        elif task_type == 1: # Testing
            model.eval()
            with torch.no_grad():
                cut_layer, target = cut_layer.to(args.device), target.to(args.device)
                output = model(cut_layer)
                is_correct = (output.argmax(dim=1) == target).long().detach().cpu()
                dist.send(tensor=is_correct, dst=client_id)

        else:
            raise ValueError(f"Server: Unknown task type {task_type}")

    if len(losses) > 0:
        plt.plot(epochs, losses)
        plt.xlabel('Processed requests')
        plt.ylabel('Average Loss')
        plt.savefig(args.plot_dir)
        print(f"Server: Loss curve saved to {args.plot_dir}")

    dist.destroy_process_group()

def run_client(rank, data, target, args):
    dist.init_process_group(backend='gloo', init_method=args.init_method, rank=rank, world_size=args.n_client + 1)
    model = MLP(5000, args.hidden_layers_client, activation=args.activation, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) if args.optimizer == 'adam' else torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    train_loader, test_loader = load_data(data, target, args.batch_size, args.test_ratio)

    for epoch in range(args.n_epoch): # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            cut_layer = model(x)

            # Send the batch size and the task type (training) to the server
            dist.send(tensor=torch.tensor([x.size(0), 0], dtype=torch.long), dst=0)
            # Send the cut layer and the label to the server
            combined2 = torch.cat((cut_layer, y.view(-1, 1).float()), dim=1).detach().cpu()
            dist.send(tensor=combined2, dst=0)

            # Receive the gradient of the cut layer from the server
            cut_layer_grad = torch.zeros_like(cut_layer).detach().cpu()
            dist.recv(tensor=cut_layer_grad, src=0)
            cut_layer_grad = cut_layer_grad.to(args.device)

            optimizer.zero_grad()
            cut_layer.backward(cut_layer_grad)
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad(): # Testing
            for x, y in test_loader:
                x, y = x.to(args.device), y.to(args.device)
                cut_layer = model(x)

                # Send the batch size and the task type (testing) to the server
                dist.send(tensor=torch.tensor([x.size(0), 1], dtype=torch.long), dst=0)
                # Send the cut layer and the label to the server
                combined2 = torch.cat((cut_layer, y.view(-1, 1).float()), dim=1).detach().cpu()
                dist.send(tensor=combined2, dst=0)

                # Receive the correctness of the prediction from the server
                is_correct = torch.zeros_like(y).detach().cpu()
                dist.recv(tensor=is_correct, src=0)
                correct += is_correct.sum().item()

        print(f"Client {rank}: Epoch {epoch + 1}, Accuracy {correct / len(test_loader.dataset)}")  
    
    dist.send(tensor=torch.tensor([0, -100], dtype=torch.long), dst=0) # Send stop signal
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda", help="device to use for training / testing")
    parser.add_argument('--init_method', type=str, default='tcp://localhost:23456', help="initialization method for distributed training")
    parser.add_argument('--n_epoch', type=int, default=40, help="number of epochs")
    parser.add_argument('--n_data', type=int, default=None, help="number of data instances to use, only specify when data_per_client is None")
    parser.add_argument('--n_data_per_client', type=int, default=None, help="number of data instances to use per client, only specify when n_data is None")
    parser.add_argument('--test_ratio', type=float, default=0.2, help="ratio of test data")
    parser.add_argument('--seed', type=int, default=42, help="random seed for data splitting")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam', help="optimizer, adam or sgd")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--activation', type=str, choices=['relu', 'tanh', 'sigmoid'], default='relu', help="activation function, relu, tanh or sigmoid")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--hidden_layers_client', type=int, nargs='+', default=[256], help="hidden layers for client")
    parser.add_argument('--hidden_layers_server', type=int, nargs='+', default=[256], help="hidden layers for server")
    parser.add_argument('--n_report', type=int, default=50, help="report every n requests")
    parser.add_argument('--n_client', type=int, default=3, help="number of clients")
    parser.add_argument('--plot_dir', type=str, default='loss.jpg', help="directory to save loss curve")
    args = parser.parse_args()

    if args.n_data is not None and args.n_data_per_client is not None:
        raise ValueError("Only one of n_data and n_data_per_client can be specified")

    print("Loading data from gisette.arff...")
    data, target = load_from_arff('gisette.arff', label_count=1)
    data = data.toarray()
    target = target.toarray()

    if args.n_data is not None:
        if args.n_data > len(data):
            print(f"Warning: n_data {args.n_data} is greater than the number of data instances {len(data)}")
        data = data[:args.n_data]
        target = target[:args.n_data]
    elif args.n_data_per_client is not None:
        if args.n_data_per_client * args.n_client > len(data):
            print(f"Warning: n_data_per_client * n_client {args.n_data_per_client * args.n_client} is greater than the number of data instances {len(data)}")
        data = data[:args.n_data_per_client * args.n_client]
        target = target[:args.n_data_per_client * args.n_client]
    
    data_splits = np.array_split(data, args.n_client)
    target_splits = np.array_split(target, args.n_client)
    target_splits = [target.reshape(-1) for target in target_splits]

    processes = []
    print("Starting server and clients...")

    p = mp.Process(target=run_server, args=(0, args))
    p.start()
    processes.append(p)

    for i in range(args.n_client):
        p = mp.Process(target=run_client, args=(i + 1, data_splits[i], target_splits[i], args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()