import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F


def train(train_data, val_data, encoder, epochs, batch_size, num_workers,
          device, uniqueID, block_size, lr):
    rates = [0.1,0.01,0.001,0.0001,0.00001]
    try:
        learningRate = rates[lr]
    except:
        print("Invalid LR")
        learningRate = 0.001
    print("Learning rate is: ", learningRate)

    # set up loss arrays and loss
    sym_losses = []
    val_losses = []
    pred = []
    targ = []

    #set up optimizers
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.001)

    best_valloss = np.inf
    for epoch in range(epochs):
        loader = DataLoader(train_data, shuffle=False,
                            batch_size=batch_size)

        print('epoch', epoch)
        running_loss_sym = []
        for i, (p, e) in enumerate(loader):
            optimizer_encoder.zero_grad(set_to_none=False)

            p = p.to(device)  # points
            e = e.to(device)  # eq array

            # run data through encoder
            logits, loss_sym, loss_sym_indiv = encoder(e, p, tokenizer=train_data.itos)

            loss_sym.backward()

            running_loss_sym.append(loss_sym.item())

            #step optimizer and zero grads
            optimizer_encoder.step()

        sym_losses.append(np.average(running_loss_sym))

        #validation
        val_loss, val_pred, val_targ = test(val_data,encoder,
                                  batch_size=32,block_size=block_size,device=device)
        val_losses.append(val_loss)
        pred.append(val_pred)
        targ.append(val_targ)
        #
        print("Epoch", epoch, "sym loss:", np.average(running_loss_sym), flush=True)
        print("Epoch", epoch, "val loss:", val_loss, flush=True)


        # Save best model by validation loss
        if val_loss < best_valloss:
            best_valloss = val_loss
            torch.save(encoder.state_dict(), 'Pretrained/encoder'+str(uniqueID)+'_best')
            print('Best Loss:',val_loss,"model saved")
    #
    #
    # print("validation pred size", np.array(pred).shape)
    # print("validation targ size", np.array(targ).shape)
    #
    np.save('data/sym' + str(uniqueID), np.array(sym_losses))
    np.save('data/val' + str(uniqueID), np.array(val_losses))
    np.save('data/pred' + str(uniqueID), np.array(pred))
    np.save('data/targ' + str(uniqueID), np.array(targ))

def test(data, encoder, batch_size, block_size, device):
    encoder.eval()
    val_loader = DataLoader(data, shuffle=True,
                            batch_size=batch_size)
    loss = []
    predicted = []
    target = []
    with torch.no_grad():
        for i, (p, e) in enumerate(val_loader):

            p = p.to(device)  # points
            e = e.to(device)  # eq array

            decoder_eqns = torch.zeros((e.shape[0], block_size))

            # run data through encoder
            logits, loss_sym, loss_sym_indiv = encoder(e, p, tokenizer=data.itos)

            # get encoder output
            for z in range(logits.size()[0]):
                decoder_eqns[z] = torch.argmax(logits[z], dim=1)

            loss.append(loss_sym.item())
            for pred in decoder_eqns:
                predicted.append(pred.numpy())
            for targ in e:
                target.append(targ.cpu().numpy())

        targ_array = np.array(target)
        pred_array = np.array(predicted)

    return np.average(loss), pred_array, targ_array



if __name__ == '__main__':
    print('hello')