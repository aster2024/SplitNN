This is a pseudo-distributed version of SplitNN. It uses **“torch.distributed”** API to deploy the model locally on different processes. Each client has a MLP truncated to a certain layer, and the server contains the remaining part of the MLP. During training, the client sends the hidden state at the cut layer and the label to the server and receives the gradient on the layer from the server. We also implemented testing process to evaluate the model's performance.

Before running the code, please download the gisette dataset (sparse arff format) from [OpenML:gisette](https://www.openml.org/data/download/18631146/gisette.arff) (It is too large to put in the attachment of the email) and put it in the same directory as the code. 

Under the default hyperparameters in the code, the validation accuracy on Client 1, 2, 3 in the last epoch are 94%, 96%, 96% accordingly. This can be reproduced by simply running `python code.py` (may exhibit minor difference).

You can also change the arguments and observe the performance. The meaning of each argument is as follows:

- **--device**: device to use for training / testing. default: "cuda"
- **--init_method**: initialization method for distributed training. default: "tcp://localhost:23456"
- **--n_epoch**: number of epochs. default: 40
- **--n_data**: number of data instances to use, only specify when data_per_client is None. default: None
- **--n_data_per_client**: number of data instances to use per client, only specify when n_data is None. default: None
- **--test_ratio**: ratio of test data. default: 0.2
- **--seed**: random seed for data splitting. default: 42
- **--learning_rate**: learning rate. default: 0.001
- **--optimizer**: optimizer, adam or sgd. default: adam
- **--batch_size**: batch size. default: 32
- **--activation**: activation function, relu, tanh or sigmoid. default: relu
- **--dropout**: dropout rate. default: 0.2
- **--hidden_layers_client**: hidden layers for client. default: [256]
- **--hidden_layers_server**: hidden layers for server. default: [256]
- **--n_report**: report every n requests. default: 50
- **--n_client**: number of clients. default: 3
- **--plot_dir**: directory to save loss curve. default: "loss.jpg"
