import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR


def train(train_data, val_data, encoder, epochs, batch_size, num_workers,
          device, uniqueID, block_size, lr, folder):
    rates = [0.01, 0.001, 0.0001, 0.00001]
    try:
        learningRate1 = rates[lr]
        learningRate2 = rates[lr]

    except:
        print("Invalid LR")
        learningRate1 = 0.001
        learningRate2 = 0.001
    print("Learning rate 1 is: ", 0.0001)
    print("Learning rate 2 is: ", 0.0001)

    # set up loss arrays and loss
    sym_losses = []
    val_losses_sym = []
    pred = []
    targ = []
    nums_increase = 0

    # set up optimizers
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.0001)
    #scheduler = StepLR(optimizer_encoder, step_size=30, gamma=0.1)

    best_valloss = np.inf
    for epoch in range(epochs):
        loader = DataLoader(train_data, shuffle=True,
                            batch_size=batch_size, drop_last=True)

        print('epoch', epoch)
        running_loss = []
        running_loss_sym = []
        running_loss_num = []
        for i, (p, e0, e1) in enumerate(loader):
            optimizer_encoder.zero_grad(set_to_none=False)

            p = p.to(device)  # points
            e0 = e0.to(device)  # input
            e1 = e1.to(device)  # output

            # run data through encoder
            logits, loss_sym = encoder(e0, e1, p, tokenizer=train_data.itos)

            # print('logits', logits)
            # print('loss', loss_sym)

            loss_sym.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer_encoder.step()
            running_loss_sym.append(loss_sym.item())

        # losses.append(np.average(running_loss))
        sym_losses.append(np.average(running_loss_sym))
        # validation
        val_loss_sym, val_pred, val_targ = test(val_data, encoder,
                                                              batch_size=10, block_size=block_size, device=device)
        val_losses_sym.append(val_loss_sym)
        pred.append(val_pred)
        targ.append(val_targ)

        # print("Epoch", epoch, "total loss:", np.average(running_loss), flush=True)
        print("Epoch", epoch, "sym loss:", np.average(running_loss_sym), flush=True)
        print("Epoch", epoch, "val loss sym:", val_loss_sym, flush=True)

        np.save('data/' + str(folder) + '/sym' + str(uniqueID), np.array(sym_losses))
        np.save('data/' + str(folder) + '/valsym' + str(uniqueID), np.array(val_losses_sym))
        np.save('data/' + str(folder) + '/pred' + str(uniqueID), np.array(pred))
        np.save('data/' + str(folder) + '/targ' + str(uniqueID), np.array(targ))


        # Save best model by validation loss            
        if epoch == epochs-1:
            torch.save(encoder.state_dict(), 'SavedModels/' + str(folder) + '/encoder' + str(uniqueID))

        #step scheduler
        #scheduler.step()



    print("validation pred size", np.array(pred).shape)
    print("validation targ size", np.array(targ).shape)
    print("best val loss", best_valloss)

    np.save('data/' + str(folder) + '/sym' + str(uniqueID), np.array(sym_losses))
    np.save('data/' + str(folder) + '/valsym' + str(uniqueID), np.array(val_losses_sym))
    np.save('data/' + str(folder) + '/pred' + str(uniqueID), np.array(pred))
    np.save('data/' + str(folder) + '/targ' + str(uniqueID), np.array(targ))


def test(data, encoder, batch_size, block_size, device):
    encoder.eval()
    val_loader = DataLoader(data, shuffle=True,
                            batch_size=batch_size)
    losses_sym = []
    predicted = []
    target = []
    with torch.no_grad():
        for i, (p, e0, e1) in enumerate(val_loader):

            p = p.to(device)  # points
            e0 = e0.to(device)  # input
            e1 = e1.to(device)  # output
            save_eqns = torch.zeros((e1.shape[0], block_size))

            # run data through encoder
            logits, loss_sym = encoder(e0, e1, p, tokenizer=data.itos)
            losses_sym.append(loss_sym.item())

            for z in range(logits.size()[0]):
                save_eqns[z] = torch.argmax(logits[z], dim=1)
            for pred in save_eqns:
                predicted.append(pred.numpy())
            for targ in e1:
                target.append(targ.cpu().numpy())

        targ_array = np.array(target)
        pred_array = np.array(predicted)

    return np.average(losses_sym), pred_array, targ_array


if __name__ == '__main__':
    print('hello')
