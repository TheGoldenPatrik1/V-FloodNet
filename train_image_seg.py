import matplotlib
matplotlib.use('Agg')
import os
import traceback
import argparse
import gc
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.losses import DiceLoss
import matplotlib.pyplot as plt
from torch.utils import data
from image_module.dataset_water import WaterDataset_RGB

ROOT_DIR = './'

DEFAULT_CHKPT_DIR = os.path.join(ROOT_DIR, 'output', 'img_seg_checkpoint')

def train(args):
    """
    Executes train script given arguments
    :param args: Training parameters
    :return:
    """
    try:
        torch.cuda.empty_cache()
    except:
        print("Error clearing cache.")
        print(traceback.format_exc())

    dataset_path = args.dataset_path
    input_shape = args.input_shape
    batch_size = args.batch_size
    init_lr = args.init_lr
    epochs = args.epochs
    out_path = args.out_path
    encoder_name = args.encoder
    model_name = args.model
    encoder_weights = args.encoder_weights
    verbose = args.verbose
    patience = args.patience

    verbose = False if verbose.lower() == 'false' else True

    # Input size must be a multiple of 32 as the image will be subsampled 5 times
    train_dataset = WaterDataset_RGB(
        mode='train_offline',
        dataset_path=dataset_path,
        input_size=(416, 416)
    )

    val_dataset = WaterDataset_RGB(
        mode='val_offline',
        dataset_path=dataset_path,
        input_size=(input_shape, input_shape)
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    model = None

    model_name = model_name.lower()
    print()
    if model_name == 'deeplabv3+':
        print(f"Loading DeepLabV3+ from SMP with weights from '{encoder_weights}'...")
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
    elif model_name == 'linknet':
        print(f"Loading Linknet from SMP with weights from '{encoder_weights}'...")
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
    else:
        print("Unrecognized or unspecified model name, exiting...")
        return

    # Train Model with given backbone
    try:
        train_model(
            model,
            init_lr=init_lr,
            num_epochs=epochs,
            out_path=out_path,
            train_loader=train_loader,
            val_loader=val_loader,
            encoder_name=encoder_name,
            model_name=model_name,
            batch_size=batch_size,
            verbose=verbose,
            patience=patience
        )
    except:
        print(traceback.format_exc())
    try:
        model = None
        gc.collect()
    except:
        print(traceback.format_exc())

def train_model(model, init_lr, num_epochs, out_path, train_loader, val_loader, encoder_name, model_name, batch_size, verbose, patience):
    """
    Trains a single image given model and further arguments
    :param model: Model from SMP library
    :param init_lr: Initial learning rate
    :param num_epochs: Number of epochs to train
    :param out_path: Folder to output checkpoints and model
    :param train_loader: Dataloader for train dataset
    :param val_loader: Dataloader for validation dataset
    :param model_name: Name of the model architecture
    :param batch_size: Batch size value
    :param verbose: Whether the logs should be verbose
    :param patience: How many epochs to wait for improvement
    :return:
    """
    plots_dir = os.path.join(out_path, 'graphs')
    checkpoints_dir = os.path.join(out_path, 'checkpoints')
    models_dir = os.path.join(out_path, 'model')

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    loss = DiceLoss()
    metrics = [
        IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=init_lr),
    ])

    print(f"Using device: {device}")

    # Create training epoch
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=verbose
    )

    # Create validation epoch
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=verbose
    )

    max_score = 0
    best_model = None
    patience_counter = 0

    train_iou_score_ls = []
    train_dice_loss_ls = []

    val_iou_score_ls = []
    val_dice_loss_ls = []

    # Go through each epoch
    for epoch in range(0, num_epochs):
        title = 'Epoch: {}'.format(epoch)
        print('\nEpoch: {}'.format(epoch))

        # Epoch logs
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        # Checkpoint to resume training
        checkpoint = {
            'epoch': epoch,
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss.state_dict()
        }

        # Get IOU score
        score = float(valid_logs['iou_score'])

        # Save checkpoint every 10 epochs
        checkpoint_savepth = os.path.join(checkpoints_dir, 'epoch_' + str(epoch).zfill(3) + '_score' + str(score) + '.pth')
        freq_save = 10
        if epoch % freq_save == 0:
            torch.save(checkpoint, checkpoint_savepth)

        # Check score on valid dataset
        if score > max_score:
            max_score = score
            new_file_name = model_name + '_' + encoder_name + '_batchsize_' + str(batch_size) + '_epoch_' + str(epoch).zfill(3) + '_score_' + str(score) + '.pth'
            model_savepth = os.path.join(models_dir, new_file_name)
            torch.save(model, model_savepth)
            print(f'New best model "{new_file_name}" detected')
            # Remove old files if they exist:
            for _f in os.listdir(models_dir):
                if _f != new_file_name and _f == best_model:
                    os.remove(os.path.join(models_dir, _f))
                    print(f'Old best model "{_f}" was deleted')
            best_model = new_file_name
            patience_counter = 0
        else:
            patience_counter += 1

        # Adjust learning rate halfway through training.
        if epoch == int(num_epochs / 2):
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decreased decoder learning rate to 1e-5!')

        train_iou_score_ls.append(train_logs['iou_score'])
        train_dice_loss_ls.append(train_logs['dice_loss'])

        val_iou_score_ls.append(valid_logs['iou_score'])
        val_dice_loss_ls.append(valid_logs['dice_loss'])

        plot_train_filepth = os.path.join(plots_dir, 'epoch_' + str(epoch).zfill(3) + '_train.png')
        plot_val_filepth = os.path.join(plots_dir, 'epoch_' + str(epoch).zfill(3) + '_val.png')
        plt.plot(train_iou_score_ls, label='train iou_score')
        plt.plot(train_dice_loss_ls, label='train dice_loss')
        plt.legend(loc="upper left")
        plt.title(title)
        plt.savefig(plot_train_filepth)
        plt.close()

        plt.plot(val_iou_score_ls, label='val iou_score')
        plt.plot(val_dice_loss_ls, label='val dice_loss')
        plt.legend(loc="upper left")
        plt.title(title)
        plt.savefig(plot_val_filepth)
        plt.close()

        if patience_counter == patience:
            print(f'No improvement after {patience} epochs; exiting early')
            break

"""
    python train_segmodel.py --dataset_path --encoder --model
"""
if __name__ == '__main__':
    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch WaterNet Model Testing')
    # Required: Path to the .pth file.
    parser.add_argument('--dataset-path',
                        type=str,
                        metavar='PATH',
                        help='Path to the dataset. Expects format shown in the header comments.')
    # Required: Encoder name.
    parser.add_argument('--encoder',
                        type=str,
                        metavar='PATH',
                        help='Encoder name, as used by segmentation_model.pytorch library')
    # Required: Which model architecture to use.
    parser.add_argument('--model',
                        default="",
                        type=str,
                        help='Model architecture to use, one of DeepLabV3+ or Linknet')
    # Optional: Image input size that the model should be designed to accept. In LinkNet, image will be
    #           subsampled 5 times, and thus must be a factor of 32.
    parser.add_argument('--input-shape',
                        default=416,
                        type=int,
                        help='(OPTIONAL) Input size for model. Single integer, should be a factor of 32.')
    # Optional: Batch size for mini-batch gradient descent. Defaults to 4, depends on GPU and your input shape.
    parser.add_argument('--batch-size',
                        default=4,
                        type=int,
                        help='(OPTIONAL) Batch size for mini-batch gradient descent.')
    # Initial Learning Rate: Initial learning rate. Learning gets set to 1e-5 halfway through training.
    parser.add_argument('--init-lr',
                        default=1e-4,
                        type=float,
                        help='(OPTIONAL) Batch size for mini-batch gradient descent.')
    # Optional: Number of epochs for training.
    parser.add_argument('--epochs',
                        default=300,
                        type=int,
                        help='(OPTIONAL) Number of epochs for training')
    # Optional: Which folder the checkpoints will be saved. Defaults to a new checkpoint folder in output.
    parser.add_argument('--out-path',
                        default=DEFAULT_CHKPT_DIR,
                        type=str,
                        metavar='PATH',
                        help='(OPTIONAL) Path to output folder, defaults to project root/output')
    # Optional: Which pre-trained weights to load for the encoder.
    parser.add_argument('--encoder-weights',
                        default='imagenet',
                        type=str,
                        help='(OPTIONAL) Pre-trained weights to load for the encoder')
    # Optional: Whether the logs should be verbose or not.
    parser.add_argument('--verbose',
                        default='true',
                        type=str,
                        help='(OPTIONAL) Whether the logs should be verbose or not')
    # Optional: The number of epochs to wait for improvement before halting the training.
    parser.add_argument('--patience',
                        default=300,
                        type=int,
                        help='(OPTIONAL) How many epochs to wait for improvement before stopping')
    _args = parser.parse_args()

    print("== System Details ==")
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print("== System Details ==")
    print()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    train(_args)
    print("Done.")