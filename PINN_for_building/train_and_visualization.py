from model import PhysicODE, transform, BuildingDataset

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


# For this example, we use cpu for simplicity.
use_cuda = False

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
Tensor = FloatTensor

def train(model, room=0, num_epochs=500, length=2, save=False, save_name=None, model_name=None):
    """
    Train a given model using the BEAR dataset.

    Args:
        model: The model to train.
        room: Room index for the dataset (default=0).
        num_epochs: Total number of training epochs (default=500).
        length: Sequence length for each training sample (default=2, repesent 15 mins, 1 unit = 15 mins, 2->current+predict, 3hr->1+12=13). 
        save: Boolean flag to save the model (default=False).
        save_name: File name to save the best model (if save=True).
        model_name: Optional identifier for the model (default=None).

    Returns:
        Trained model.
    """
    # Define the time steps for the sequence length
    t = torch.linspace(0, length, length)

    # Print sequence length for debugging
    print(length)

    # Load the training and validation datasets
    dataset1 = BuildingDataset(file='control_data.npz', N_train=7, room=room, length=length, resolution=3, train=True)
    dataset2 = BuildingDataset(file='control_data.npz', N_train=7, room=room, length=length, resolution=3, train=False)

    # Create data loaders for batching and shuffling
    dataloader = DataLoader(dataset1, batch_size=128, shuffle=False)
    val_loader = DataLoader(dataset2, batch_size=256, shuffle=False)

    # Set up optimizer, scheduler, and loss function
    lr = 1e-1  # Initial learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)  # Learning rate scheduler
    loss_fun = torch.nn.MSELoss()  # Mean Squared Error loss

    # Initialize variables to track the best model and training progress
    min_loss = 100000
    Val_Loss = []  # Store validation loss for each epoch
    Loss = []  # Store training loss for each epoch
    val_loss, cnt = 100, 0  # Initialize validation loss and patience counter

    # Start timing the training process
    start = time.time()

    ##################################################################
    # Training loop for the specified number of epochs
    for epoch in range(num_epochs):
        # Adjust learning rate manually at specific epochs for fine-tuning
        if epoch == 500:
            optimizer.param_groups[0]['lr'] = 1e-2
        elif epoch == 2000:
            optimizer.param_groups[0]['lr'] = 1e-3

        train_loss = 0.0  # Initialize training loss for this epoch

        # Training phase
        for x0, y, truex in dataloader:
            # Permute dimensions to match the expected input shape
            y = y.permute(1, 0, 2)
            truex = truex.permute(1, 0, 2)

            # Forward pass
            predx = model(x0, y, t, name=model_name)  # Model prediction

            # Compute loss
            loss = loss_fun(predx * 10, truex * 10)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Accumulate loss

        train_loss /= len(dataloader)  # Average training loss for this epoch

        # Save the best model if the current training loss is the minimum
        if save and train_loss < min_loss:
            best_model = model
            torch.save(best_model.state_dict(), save_name)
            min_loss = train_loss

        Loss.append(train_loss)  # Record training loss

        # Validation phase
        model.eval()
        pre_loss = val_loss  # Store the previous validation loss
        val_loss = 0.0  # Initialize validation loss for this epoch

        with torch.no_grad():  # Disable gradient calculation for validation
            for x0, y, truex in val_loader:
                y = y.permute(1, 0, 2)
                truex = truex.permute(1, 0, 2)
                predx = model(x0, y, t, name=model_name)
                loss = loss_fun(predx * 10, truex * 10)
                val_loss += loss.item()

        val_loss /= len(val_loader)  # Average validation loss

        # Adjust learning rate or stop training if validation loss doesn't improve
        if val_loss > pre_loss:
            cnt += 1
            if cnt > 100:  # Stop early if no improvement for 100 epochs
                break
            elif cnt > 10:  # Halve learning rate if no improvement for 10 epochs
                optimizer.param_groups[0]['lr'] /= 2
        else:
            cnt = 0  # Reset patience counter if validation loss improves

        Val_Loss.append(val_loss)  # Record validation loss

        # Print progress every 50 epochs
        if epoch % 50 == 0:
            print(f'Epoch = {epoch}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time Used: {np.round(time.time()-start, 2)}s')

    # Save loss data for future analysis
    np.savez(f'{save_name}_{room}_loss.npz', loss=np.array(Loss), val_loss=np.array(Val_Loss))

    return model  # Return the trained model


def main():
    e=4000
    ## 3 hr horizon (length=13)  current 1 + predict 12 = 13
    torch.manual_seed(1234)
    model_P_room_2_hori_3h = PhysicODE()
    model_P_room_2_hori_3h=train(model_P_room_2_hori_3h,room=2,num_epochs=e,length=13,save=True,save_name=f'checkpoint/p_room_2_hori_3h.pth',model_name='P1')
    
    torch.manual_seed(1234)
    model_C_room_2_hori_3h = PhysicODE()
    model_C_room_2_hori_3h=train(model_C_room_2_hori_3h,room=2,num_epochs=e,length=13,save=True,save_name=f'checkpoint/c_room_2_hori_3h.pth',model_name='C1')
    
    torch.manual_seed(1234)
    model_D_room_2_hori_3h = PhysicODE()
    model_D_room_2_hori_3h=train(model_D_room_2_hori_3h,room=2,num_epochs=e,length=13,save=True,save_name=f'checkpoint/d_room_2_hori_3h.pth',model_name='D1')
    
    ## 15 mins horizon (length=2, by default)
    torch.manual_seed(1234)
    model_P_room_2_hori_15m = PhysicODE()
    model_P_room_2_hori_15m = train(model_P_room_2_hori_15m,room=2,num_epochs=e,save=True,save_name=f'checkpoint/p_room_2_hori_15.pth',model_name='P1')
    
    torch.manual_seed(1234)
    model_C_room_2_hori_15m = PhysicODE()
    model_C_room_2_hori_15m=train(model_C_room_2_hori_15m,room=2,num_epochs=e,save=True,save_name=f'checkpoint/c_room_2_hori_15.pth',model_name='C1')
    
    torch.manual_seed(1234)
    model_D_room_2_hori_15m = PhysicODE()
    model_D_room_2_hori_15m=train(model_D_room_2_hori_15m,room=2,num_epochs=e,save=True,save_name=f'checkpoint/d_room_2_hori_15.pth',model_name='D1')
    
    ######################################
    # Training complete, visualize the results.
    ######################################
    
    model_names = ['D1', 'P1', 'C1']
    model_names2 = ['NN', 'PINN', 'PINN+NN']
    colors = ['#ff3534', '#008000', '#4c4cff', '#ef857e'] ##4c4cff
    alphas = [1, 1, 1]
    lims = [68, 79]
    fig, ax = plt.subplots(2, 1, figsize=(8, 3))
    room = 2

    model_set_1 = [model_D_room_2_hori_15m, model_P_room_2_hori_15m, model_C_room_2_hori_15m]
    model_set_2 = [model_D_room_2_hori_3h, model_P_room_2_hori_3h, model_C_room_2_hori_3h]

    for i, length in enumerate([2, 13]):
        dataset = BuildingDataset(file='control_data.npz', room=room, N_train=14, length=length, resolution=3, train=False)
        dataloader = DataLoader(dataset, batch_size=20000, shuffle=False)

        for j in range(3):
            model_name = model_names[j]
            model = model_set_1[j] if length == 2 else model_set_2[j]

            t = torch.linspace(0, length, length)
            for x0, y, truex in dataloader:
                y = y.permute(1, 0, 2)  # Ensure proper shape: (L, B, features)
                truex = truex.permute(1, 0, 2).to(device)
                predx = model(x0.to(device), y.to(device), t.to(device), name=model_name)  # Output: (L, B, features)
                break  # Use the first batch

            predx = predx[:truex.shape[0], :, :].cpu().detach().numpy()  # Match sequence lengths
            truex = truex.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            # Ensure consistent lengths for transform()
            transformed_truex = transform(truex, day=14, length=length)
            transformed_predx = transform(predx, day=14, length=length)

            # Align lengths for plotting
            min_len = min(len(transformed_truex), len(transformed_predx))
            transformed_truex = transformed_truex[:min_len]
            transformed_predx = transformed_predx[:min_len]

            if j == 0:
                ax[i].plot(transformed_truex, color='black', label='Groundtruth', linewidth=1.5)

            ax[i].plot(
                np.arange(672, 672 + len(transformed_predx[672:])),  # Match x length with y
                transformed_predx[672:],
                color=colors[j],
                alpha=alphas[j],
                linewidth=1.2 - 0.1 * j * (i == 0),
                label=model_names2[j]
            )

            print(model_names2[j], np.mean((transformed_predx - transformed_truex) ** 2))

        ax[i].set_xticks([])
        ax[i].set_ylim(lims)
        ax[i].tick_params(labelsize=14)

    # Add legend and labels
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=5)
    fig.text(0.06, 0.5, 'Zone Temperature (F)', va='center', rotation='vertical', fontsize=14)
    fig.text(0.91, 0.5, 'Zone 108', va='center', rotation='vertical', fontsize=12)

    # Add x-ticks for the last subplot
    tick_positions = list(range(96, 96 * 14, 96 * 2))
    tick_labels = ['06/01', '06/03', '06/05', '06/07', '06/09', '06/11', '06/13']
    ax[-1].set_xticks(tick_positions)
    ax[-1].set_xticklabels(tick_labels, fontsize=14)
    plt.savefig('predict.png', dpi=300, bbox_inches='tight')

    

if __name__ == "__main__":
    torch.manual_seed(1234)
    main()