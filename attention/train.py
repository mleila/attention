import time
from random import random
import torch

from attention.constants import DECODER_INPUT, ENCODER_INPUT, TRAIN, VALID



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


def train_simpleRNN_batch(
    input_batch,
    target_batch,
    encoder,
    decoder,
    encoder_opt,
    decoder_opt,
    criterion,
    device,
    use_teacher_forcing=True
    ):
    """
    TODO: Implement teacher forcing
    """
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    batch_size, seq_size = input_batch.shape
    hidden = encoder.initHidden(batch_size=batch_size, device=device)

    encoder_outputs = torch.zeros(seq_size, batch_size, encoder.hidden_size)
    decoder_outputs = torch.zeros(seq_size, batch_size, decoder.output_size)

    for i in range(seq_size):
        tokens = input_batch[:, i]
        output, hidden = encoder(tokens, hidden)
        encoder_outputs[i] = output

    for i in range(seq_size-1):
        decoder_input = target_batch[:, i].unsqueeze(0)
        if use_teacher_forcing and random() <= 0.5 and i > 0:
            prev_output = torch.argmax(decoder_outputs[i-1], dim=-1).unsqueeze(0)
            output, hidden = decoder(prev_output, hidden)
        else:
            output, hidden = decoder(decoder_input, hidden)
        decoder_outputs[i] = output
        loss = criterion(output, target_batch[:, i+1])
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()
        hidden = hidden.detach()

    return loss


class Translation_Trainer:
    """
    # todo: adapt this for the encoder-decoder architecture
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
            train_losses = []

            # save model training checkpoint
            if checkpoint:
                training_state = self.get_training_state()
                timestamp = int(time.time())
                filename = f"{self.model_dir}/{timestamp}.pth.tar"
                save_checkpoint(training_state, filename=filename)

            # set training mode
            self.model.train()
            dataset.set_split(TRAIN)

            # training loop
            for batch_gen in self.data_loader(dataset, batch_size=batch_size):
                encoder_input, decoder_input = batch_gen[ENCODER_INPUT], batch_gen[DECODER_INPUT]
                self.optimizer.zero_grad()
                decoder_outputs = self.model(encoder_input, decoder_input)
                loss = self.loss_func(decoder_outputs, decoder_input[:, 1:])
                loss_batch = loss.item()
                train_losses.append(loss_batch)
                loss.backward()
                self.optimizer.step()

            avg_loss = sum(train_losses)/len(train_losses)
            print(f'Completed Epoch {epoch} with average training loss of {avg_loss:.2f}')

            # set validation mode
            self.model.eval()
            dataset.set_split(VALID)
            french_vocab = dataset.vectorizer.french_vocab
            sos_token = french_vocab.lookup_token(french_vocab.sos)

            # validation loop
            valid_losses = []
            for batch_gen in self.data_loader(dataset, batch_size=batch_size):
                encoder_input, decoder_input = batch_gen[ENCODER_INPUT], batch_gen[DECODER_INPUT]
                decoder_outputs = self.model(encoder_input, sos_token=sos_token)
                loss = self.loss_func(decoder_outputs, decoder_input[:, 1:])
                loss_batch = loss.item()
                valid_losses.append(loss_batch)
            avg_loss = sum(valid_losses)/len(valid_losses)
            print(f'Completed Epoch {epoch} with average validation loss of {avg_loss:.2f}')
