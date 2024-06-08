import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from tqdm.notebook import tqdm as tqdm_nb
from sklearn.metrics import r2_score

def reset_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class MLP(torch.nn.Module):
    def __init__(self, layer_sizes, activation=torch.nn.ReLU, dropout=0):
        super(MLP, self).__init__()

        # Set seed for reproducibility
        reset_seeds()

        self.layers = torch.nn.ModuleList()

        for i, (l1, l2) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.layers.append(torch.nn.Linear(l1, l2))
            if i < len(layer_sizes)-2:
                self.layers.append(torch.nn.Dropout(dropout))
                self.layers.append(activation())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_weights(self):
        return np.hstack([p.data.clone().cpu().detach().numpy().flatten() for p in self.parameters()])


class SimpleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()

        # Set seed for reproducibility
        reset_seeds()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)

    def get_weights(self):
        return self.linear.weight.clone().cpu().detach().numpy().flatten()


def plot_training_visualisation(histories, l1_line=True, l2_line=True, alpha=0.025, savename=None):
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(12, 5), sharex=True, dpi=100)
    plt.subplots_adjust(hspace=0, wspace=0)

    ax1.plot(histories["train_loss"], label="Train loss")
    ax1.plot(histories["val_loss"], label="Validation loss")
    best_epoch_str = "\n".join([
        f"Best train loss: {np.min(histories['train_loss']):.4f}",
        f"Best validation loss: {np.min(histories['val_loss']):.4f}"
    ])
    ax1.text(0.02, 0.05, best_epoch_str,
        ha='left', va='bottom', transform=ax1.transAxes)
    best_epoch = np.argmin(histories["val_loss"])
    ax1.axvline(best_epoch, color='r', dashes=[20,10], lw=1)
    ax2.axvline(best_epoch, color='r', dashes=[20,10], lw=1)
    ax1.set(xlabel="Epoch", ylabel="Loss", yscale="log", xscale="log")
    ax1.set(ylim=[v*f for v, f in zip(ax1.get_ylim(), [1.0, 2.0])])
    ax1.text(best_epoch*1.1, 0.98, f"Best epoch", va='top', ha='left', transform=ax1.get_xaxis_transform())
    ax1.grid(which='major', linestyle='-', linewidth=0.5)
    ax1.grid(which='minor', dashes=[20,10], linewidth=0.3)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)

    num_params = histories["model_state"].shape[1]
    for i in range(num_params):
        l = ax2.plot(histories["model_state"][:, i], alpha=alpha, color='k')
    if l1_line or l2_line:
        twin_ax2 = ax2.twinx()
    if l1_line:
        twin_ax2.plot(np.mean(np.abs(histories["model_state"]), axis=1), label="L1 norm", color='r')
    if l2_line:
        twin_ax2.plot(np.mean(np.square(histories["model_state"]), axis=1), label="L2 norm", color='b')

    ax2.set(xlabel="Epoch", ylabel="Parameter values", yscale="linear", xscale="log")
    ax2.grid(which='major', linestyle='-', linewidth=0.5)
    ax2.grid(which='minor', dashes=[20,10], linewidth=0.3)
    if l1_line or l2_line:
        twin_ax2.set_ylim([v*f for v, f in zip(twin_ax2.get_ylim(), [1.0, 1.2])])
        twin_ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1, frameon=False)
        twin_ax2.set(ylabel="$L_p$ norm")

    if savename:
        plt.savefig(f"figures/{savename}", dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions_vs_truth(model, train_dataset, val_dataset, device=None, savename=None):
    if device is None:
        device = "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        train_sample_x, train_sample_y = train_dataset[:]
        train_sample_x, train_sample_y = train_sample_x.to(device), train_sample_y.to(device)
        train_predictions = model(train_sample_x).cpu().detach().numpy()

        val_sample_x, val_sample_y = val_dataset[:]
        val_sample_x, val_sample_y = val_sample_x.to(device), val_sample_y.to(device)
        val_predictions = model(val_sample_x).cpu().detach().numpy()

    train_r2 = r2_score(train_sample_y, train_predictions)
    val_r2 = r2_score(val_sample_y, val_predictions)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 5), sharey=True, dpi=100)
    plt.subplots_adjust(hspace=0, wspace=0)

    ax1.scatter(train_sample_y, train_predictions)
    ax1.plot([train_sample_y.min(), train_sample_y.max()], [train_sample_y.min(), train_sample_y.max()], 'k--')
    ax1.set_xlabel('True values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Training set')
    ax1.text(0.05, 0.95, fr"$R^2 ={train_r2:.4f}$", ha='left', va='top', transform=ax1.transAxes)

    ax2.scatter(val_sample_y, val_predictions)
    ax2.plot([val_sample_y.min(), val_sample_y.max()], [val_sample_y.min(), val_sample_y.max()], 'k--')
    ax2.set_xlabel('True values')
    ax2.set_yticklabels([])
    ax2.set_title('Validation set')
    ax2.text(0.05, 0.95, fr"$R^2 ={val_r2:.4f}$", ha='left', va='top', transform=ax2.transAxes)

    if savename:
        plt.savefig(f"figures/{savename}", dpi=300, bbox_inches='tight')
    plt.show()


def training_loop(model,
                  train_dataset, validation_dataset,
                  weight_decay=0, l2_reg=0, l1_reg=0,
                  batch_size=None, lr=0.01, num_epochs=2_500,
                  criterion=torch.nn.MSELoss(),
                  device=None,
                  optimiser="adam"):

    # Set seed for reproducibility
    reset_seeds()

    if device is None:
        device = "cpu"
    model.to(device)

    # Record training history
    histories = defaultdict(list)

    # Set up batch sizes and loader
    batch_size = min(batch_size, len(train_dataset)) if batch_size else len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False, drop_last=True)

    # Set up optimiser and loss function
    if optimiser == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimiser == "adamw":
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimiser == "sgd":
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimiser not recognised.")
    criterion = criterion.to(device)

    # Record best epoch
    best_model_state = None
    best_loss = np.inf

    # Starting point
    with torch.no_grad():
        histories["model_state"].append(model.get_weights())
        train_loss = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            train_loss.append(criterion(model(x), y).item())
        histories["train_loss"].append(np.mean(train_loss))
        val_loss = []
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            val_loss.append(criterion(model(x), y).item())
        histories["val_loss"].append(np.mean(val_loss))

    # Training loop
    for epoch in tqdm_nb(range(num_epochs)):
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()

            # Manual regularisation
            if l2_reg:
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad.data += l2_reg*param.data
            if l1_reg:
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad.data += l1_reg*torch.sign(param.data)

            optimiser.step()
            epoch_losses.append(loss.item())

        # Record batch
        histories["train_loss"].append(np.mean(epoch_losses))
        histories["model_state"].append(model.get_weights())

        # Record validation loss
        model.eval()
        epoch_losses = []
        with torch.no_grad():
            for x, y in validation_loader:
                x, y = x.to(device), y.to(device)
                val_loss = criterion(model(x), y)
                epoch_losses.append(val_loss.item())

        histories["val_loss"].append(np.mean(epoch_losses))

        if histories["val_loss"][-1] < best_loss:
            best_loss = histories["val_loss"][-1]
            best_model_state = model.state_dict()

    # Convert to numpy arrays
    for key, value in histories.items():
        histories[key] = np.array(value)

    # Load best model
    model.load_state_dict(best_model_state)
    model.to("cpu")
    histories["model"] = model

    return histories