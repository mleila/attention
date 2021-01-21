from attention.constants import DECODER_INPUT, ENCODER_INPUT
import time
import torch


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])


def get_latest_model_checkpoint(model_dir):
    checkpoints = [f for f in model_dir.glob('**/*') if f.is_file()]
    if not checkpoints:
        return
    checkpoints = sorted(checkpoints)
    return checkpoints[-1]


class Translation_Trainer:
    """
    This class is responsible for training models and all the associated bookkeeping
    """

    def __init__(self, data_loader, optimizer, model, model_dir, loss_func, device):
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.model = model
        self.model_dir = model_dir
        self.loss_func = loss_func
        self.device = device


    def get_training_state(self):
        return {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
            }


    def run(self, nb_epochs, dataset, batch_size, checkpoint=None):

        if checkpoint:
            latest_checkpoint = get_latest_model_checkpoint(self.model_dir)
            if latest_checkpoint:
                map_location = torch.device(self.device)
                checkpoint_state = torch.load(latest_checkpoint, map_location=map_location)
                load_checkpoint(checkpoint_state, self.model)

        for epoch in range(nb_epochs):
            losses = []

            # save model training checkpoint
            if checkpoint:
                training_state = self.get_training_state()
                timestamp = int(time.time())
                filename = f"{self.model_dir}/{timestamp}.pth.tar"
                save_checkpoint(training_state, filename=filename)

            for batch_gen in self.data_loader(dataset, batch_size=batch_size):
                encoder_input, decoder_input = batch_gen[ENCODER_INPUT], batch_gen[DECODER_INPUT]
                self.optimizer.zero_grad()
                decoder_outputs = self.model(encoder_input, decoder_input)
                loss = self.loss_func(decoder_outputs, decoder_input[:, 1:])
                loss_batch = loss.item()
                losses.append(loss_batch)
                loss.backward()
                self.optimizer.step()

            avg_loss = sum(losses)/len(losses)
            print(f'Completed Epoch {epoch} with average loss of {avg_loss:.2f}')
