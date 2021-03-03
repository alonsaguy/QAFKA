from torch.utils.data import DataLoader
import torch

class Trainer():
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, early_stopping=50, print_every=1, **kw):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: Train and test losses per epoch.
        """
        train_loss, val_loss = [], []
        best_loss = None
        best_model = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            print(f'--- EPOCH {epoch + 1}/{num_epochs} ---')

            loss = self.train_epoch(dl_train, **kw)
            train_loss.append(loss)

            loss = self.test_epoch(dl_test, **kw)
            val_loss.append(loss)

            if(epoch == 0):
                best_loss = loss
            else:
                if(loss >= best_loss):
                    epochs_without_improvement += 1
                    if(epochs_without_improvement > early_stopping):
                        print("Reached early stopping criterion")
                        self.model.load_state_dict(torch.load('best_model'))
                        break
                else:
                    epochs_without_improvement = 0
                    best_loss = loss
                    torch.save(self.model.state_dict(), 'best_model')

            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print("Epoch", epoch + 1, ": Train loss =", train_loss[-1].item())
                print("Epoch", epoch + 1, ": Validation loss =", val_loss[-1].item())


    def train_epoch(self, dl_train: DataLoader, **kw):
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()
        total_loss = 0
        cnt = 0
        for X_train, y_train in dl_train:
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(X_train)

            # Compute Loss
            loss = self.loss_fn(y_pred.squeeze(), y_train)
            total_loss += loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            cnt += 1
        return total_loss / cnt

    def test_epoch(self, dl_test: DataLoader, **kw):
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()
        total_loss = 0
        cnt = 0
        for X_test, y_test in dl_test:
            # Forward pass
            y_pred = self.model(X_test)
            total_loss += self.loss_fn(y_pred.squeeze(), y_test)
            cnt += 1

        return total_loss / cnt
