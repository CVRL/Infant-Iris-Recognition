import os
import sys
import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader, ConcatDataset, random_split
from modules.dataset import IrisDataset
from modules.networks import NestedSharedAtrousResUNet
from modules.criterion import CompleteDiceLoss
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import madgrad
from modules.metrics import iou_score
from torch.autograd import Variable
import torch.multiprocessing as mp


# Set the start method to 'spawn' to avoid CUDA initialization issues with multiprocessing
# mp.set_start_method('spawn', force=True)

# Set device (GPU or CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define input and target transformations
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

target_transform = transforms.Compose([
    transforms.ToTensor()
])


def plot_augmented_train_data(args):
    """
    Visualize augmented training data.

    Args:
    - args (argparse.Namespace): Arguments containing various parameters.
    """
    print("Visualization mode")

    # Initialize IrisDataset for training data
    dataset_mateusz = IrisDataset(
        image_dir=args.train_image_dir_mateusz,
        mask_dir=args.train_mask_dir_mateusz,
        input_transform=input_transform,
        target_transform=target_transform,
        mode=args.mode,
        aug_num_repetitions=args.aug_num_repetitions,
        pupil_pixel_range=args.pupil_pixel_range,
        circle_model_path=args.circle_model_path,
        circle_model_name=args.circle_model_name
    )

    dataset_openeds = IrisDataset(
        image_dir=args.train_image_dir_openeds,
        mask_dir=args.train_mask_dir_openeds,
        input_transform=input_transform,
        target_transform=target_transform,
        mode=args.mode,
        aug_num_repetitions=args.aug_num_repetitions,
        pupil_pixel_range=args.pupil_pixel_range,
        circle_model_path=args.circle_model_path,
        circle_model_name=args.circle_model_name
    )

    # # Take 50% data from openeds dataset
    # dataset_length = len(dataset_openeds)
    # lengths = [int(dataset_length * 0.5), dataset_length - int(dataset_length * 0.5)]
    # train_subset_openeds, remaining_subset_openeds = random_split(dataset_openeds, lengths, generator=torch.Generator().manual_seed(42))

    # Combine datasets
    train_dataset = ConcatDataset([dataset_mateusz, dataset_openeds])

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    # Get a single batch
    input_images, target_masks = next(iter(train_dataloader))

    # Reshape input images and target masks
    bs, augs, c, h, w = input_images.size()
    input_images = input_images.view(-1, c, h, w)
    target_masks = target_masks.view(-1, c, h, w)

    # Normalize the images
    normalized_images = (input_images - input_images.min()) / (input_images.max() - input_images.min())
    normalized_masks = target_masks / target_masks.max()

    # Concatenate images and masks along the width dimension for plotting
    batch = torch.cat((normalized_images, normalized_masks), dim=3)

    # Create a grid of images
    grid = utils.make_grid(batch, nrow=8, padding=1, normalize=False)

    # Convert torch.Tensor to numpy array
    grid = grid.permute(1, 2, 0).detach().cpu().numpy()

    # Plot the grid
    plt.figure(figsize=(20, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('training_sample.pdf', format='pdf', dpi=500, bbox_inches='tight')


def train(args, model):
    """
    Train the model.

    Args:
    - args (argparse.Namespace): Arguments containing various parameters.
    - model: Model to be trained.
    """
    print("Training mode")

    if args.log_txt:
        sys.stdout = open('./' + args.seg_model_name.lower() + '_' + str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')

    print(model)

    directory = os.path.dirname(args.seg_model_name.lower() + '_' + str(args.width) + '_' + args.tag + '_checkpoint/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize IrisDataset for training data
    dataset_mateusz = IrisDataset(
        image_dir=args.train_image_dir_mateusz,
        mask_dir=args.train_mask_dir_mateusz,
        input_transform=input_transform,
        target_transform=target_transform,
        mode=args.mode,
        aug_num_repetitions=args.aug_num_repetitions,
        pupil_pixel_range=args.pupil_pixel_range,
        circle_model_path=args.circle_model_path,
        circle_model_name=args.circle_model_name
    )

    dataset_openeds = IrisDataset(
        image_dir=args.train_image_dir_openeds,
        mask_dir=args.train_mask_dir_openeds,
        input_transform=input_transform,
        target_transform=target_transform,
        mode=args.mode,
        aug_num_repetitions=args.aug_num_repetitions,
        pupil_pixel_range=args.pupil_pixel_range,
        circle_model_path=args.circle_model_path,
        circle_model_name=args.circle_model_name
    )

    # Take 50% data from openeds dataset
    # dataset_length = len(dataset_openeds)
    # lengths = [int(dataset_length * 0.5), dataset_length - int(dataset_length * 0.5)]
    # train_subset_openeds, remaining_subset_openeds = random_split(dataset_openeds, lengths, generator=torch.Generator().manual_seed(42))

    # Combine datasets
    train_dataset = ConcatDataset([dataset_mateusz, dataset_openeds])

    # Create DataLoader for training dataset
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    # Initialize IrisDataset for validation data
    val_dataset = IrisDataset(image_dir=args.test_image_dir_newborn,
                              mask_dir=args.test_mask_dir_newborn,
                              input_transform=input_transform,
                              target_transform=target_transform)

    # Create DataLoader for validation dataset
    val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size*args.aug_num_repetitions, shuffle=False)

    # Print the length of training and validation dataloaders
    print('Training length:', len(train_dataloader))
    print('Validation length:', len(val_dataloader))

    # Choose the appropriate loss function based on the specified loss type
    if args.loss_type == 'cross_entropy':
        # Use binary cross-entropy loss
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.loss_type == 'cross_entropy+dice':
        # Use a combination of binary cross-entropy and Dice loss
        criterion = torch.nn.BCEWithLogitsLoss()
        criterion2 = CompleteDiceLoss()  # Assuming CompleteDiceLoss is a custom implementation
    elif args.loss_type == 'dice':
        # Use Dice loss
        criterion = CompleteDiceLoss()  # Assuming CompleteDiceLoss is a custom implementation
    else:
        # If the specified loss type is invalid, print an error message and exit
        print('Please select a valid loss type.')
        exit()

    # Initialize the optimizer based on the solver_name argument
    if args.solver_name == 'madgrad':
        optimizer = madgrad.MADGRAD(model.parameters(), lr=args.lr)
    elif args.solver_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    elif args.solver_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    else:
        raise ValueError("Invalid solver_name. Please choose from 'madgrad', 'Adam', or 'SGD'.")

    # If CUDA is enabled, initialize a gradient scaler for mixed-precision training
    if args.cuda:
        scaler = torch.cuda.amp.GradScaler()

    # Initialize the best validation loss to positive infinity
    best_val_loss_average = float('inf')

    # Iterate over each epoch in the specified number of epochs
    for epoch in range(1, args.num_epochs + 1):
        # Set the model to training mode
        model.train()
        # Initialize a list to store the loss for each epoch
        epoch_loss = []
        # Initialize the training IoU
        train_IoU = 0

        # Iterate over each batch in the training dataloader
        for batch, (images, labels) in enumerate(train_dataloader):
            # Reshape input images and target masks from 5D to 4D tensor
            bs, augs, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            labels = labels.view(-1, c, h, w)

            # Move data to GPU if available
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            # Wrap tensors in Variable for automatic differentiation
            inputs = Variable(images)
            targets = Variable(labels)

            # Apply label smoothing if enabled
            if args.label_smoothing:
                rand_1 = torch.rand(targets.shape)
                rand_0 = torch.rand(targets.shape)
                if args.cuda:
                    rand_1 = rand_1.cuda()
                    rand_0 = rand_0.cuda()
                soft_targets_1 = targets - rand_1 * 0.2
                soft_targets_0 = targets + rand_0 * 0.2
                targets = torch.where(targets > 0.5, soft_targets_1, soft_targets_0).requires_grad_(False)
                if args.cuda:
                    targets = targets.cuda()

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            if args.cuda:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    # Calculate loss
                    if args.loss_type == 'cross_entropy+dice':
                        loss = 0.5 * criterion(outputs, targets.reshape(outputs.shape)) + 0.5 * criterion2(outputs,
                                                                                                           targets.reshape(
                                                                                                               outputs.shape))
                    else:
                        loss = criterion(outputs, targets.reshape(outputs.shape))
            else:
                outputs = model(inputs)
                # Calculate loss
                if args.loss_type == 'cross_entropy+dice':
                    loss = 0.5 * criterion(outputs, targets.reshape(outputs.shape)) + 0.5 * criterion2(outputs,
                                                                                                       targets.reshape(
                                                                                                           outputs.shape))
                else:
                    loss = criterion(outputs, targets.reshape(outputs.shape))

            # Backward pass and optimization step
            if args.cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Compute IoU (Intersection over Union)
            IoU = iou_score(outputs.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]),
                            targets.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[3]))

            # Accumulate IoU score for the current batch
            train_IoU += IoU

            # Append loss to epoch_loss
            epoch_loss.append(loss.item())

            # Log the training loss and IoU
            if batch % args.log_batch == 0:
                # Calculate the average training loss for the logged batches
                train_loss_average = sum(epoch_loss) / len(epoch_loss)

                # Print the training loss and IoU
                print("Train loss: {aver} (epoch: {epoch}, batch: {batch}, IoU: {IoU})".format(
                    aver=train_loss_average, epoch=epoch, batch=batch, IoU=train_IoU / args.log_batch))

                # Reset the training IoU for the next set of batches
                train_IoU = 0

                # Optionally log to a text file
                if args.log_txt:
                    # Close the standard output to redirect prints to the text file
                    sys.stdout.close()

                    # Open the text file in append mode
                    sys.stdout = open('./' + args.seg_model_name.lower() + '_' +
                                      str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')

        # Initialize a list to store the validation loss for each epoch
        val_epoch_loss = []
        # Initialize the validation IoU
        val_IoU = 0
        # Set the model to evaluation mode
        model.eval()

        # Disable gradient computation during validation
        with torch.no_grad():
            # Iterate over each batch in the validation dataloader
            for batch, (images, labels) in enumerate(val_dataloader):
                # Move data to GPU if available
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                # Wrap tensors in Variable for automatic differentiation
                inputs = Variable(images)
                targets = Variable(labels)

                # Forward pass
                outputs = model(inputs)

                # Calculate validation loss
                if args.loss_type == 'cross_entropy+dice':
                    val_loss = 0.5 * criterion(outputs, targets.reshape(outputs.shape)) + \
                               0.5 * criterion2(outputs, targets.reshape(outputs.shape))
                else:
                    val_loss = criterion(outputs, targets.reshape(outputs.shape))

                # Append loss to val_epoch_loss
                val_epoch_loss.append(val_loss.item())

                # Compute IoU
                IoU = iou_score(outputs.clone().detach().requires_grad_(False).reshape(outputs.shape[0],
                                outputs.shape[2], outputs.shape[3]),
                                targets.clone().detach().requires_grad_(False).reshape(outputs.shape[0],
                                outputs.shape[2], outputs.shape[3]))
                # Accumulate IoU score for the current batch
                val_IoU += IoU

            # Calculate average validation loss and IoU
            val_loss_average = sum(val_epoch_loss) / len(val_epoch_loss)
            val_IoU /= len(val_dataloader)
            print("Val loss: {aver} (epoch: {epoch}), Val IoU: {val_IoU}".format(aver=val_loss_average,
                                                                                 epoch=epoch,
                                                                                 val_IoU=val_IoU))

            # Optionally log to a text file
            if args.log_txt:
                sys.stdout.close()
                sys.stdout = open('./' + args.seg_model_name.lower() + '_' +
                                  str(args.width) + '_' + args.tag + '_trainer_output.txt', 'a')

            # Save the model checkpoint if the validation loss has improved
            if val_loss_average < best_val_loss_average:
                best_val_loss_average = val_loss_average
                filename = os.path.join(directory, "{model}-{epoch:03}-{val}-maskIoU-{val_IoU}.pth".format(model=args.seg_model_name,
                                                                                                           epoch=epoch,
                                                                                                           val=round(val_loss_average, 6),
                                                                                                           val_IoU=round(val_IoU, 6)))
                # Check if multiple GPUs are used
                if args.multi_gpu:
                    # Save the state dictionary of the underlying model (model.module)
                    torch.save(model.module.state_dict(), filename)
                else:
                    # Save the state dictionary of the model
                    torch.save(model.state_dict(), filename)

    # Close the text file if logging to a text file
    if args.log_txt:
        sys.stdout.close()


def evaluate(args, model):
    """Evaluate the model."""
    print("Evaluation mode")

    # Initialize IrisDataset for validation data
    val_dataset = IrisDataset(image_dir=args.test_image_dir_newborn,
                              mask_dir=args.test_mask_dir_newborn,
                              input_transform=input_transform,
                              target_transform=target_transform)

    # Create DataLoader for validation dataset
    val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers,
                                batch_size=args.batch_size * args.aug_num_repetitions, shuffle=False)

    print('validation length:', len(val_dataloader))

    val_epoch_loss = []
    val_IoU = 0
    model.eval()

    # Disable gradient computation during validation
    with torch.no_grad():
        # Iterate over each batch in the validation dataloader
        for batch, (images, labels) in enumerate(val_dataloader):
            # Move data to GPU if available
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            # Wrap tensors in Variable for automatic differentiation
            inputs = Variable(images)
            targets = Variable(labels)

            # Forward pass
            outputs = model(inputs)

            # Compute IoU
            IoU = iou_score(outputs.clone().detach().requires_grad_(False).reshape(outputs.shape[0],
                            outputs.shape[2], outputs.shape[3]),
                            targets.clone().detach().requires_grad_(False).reshape(outputs.shape[0],
                            outputs.shape[2], outputs.shape[3]))

            # Accumulate IoU score for the current batch
            val_IoU += IoU

        # Compute the average Intersection over Union (IoU) for the validation dataset
        val_IoU /= len(val_dataloader)

        # Print the average IoU for the validation dataset
        print("Val IoU: {val_IoU}".format(val_IoU=val_IoU))


def main(args):
    """Main function to handle visualization, training and evaluation."""
    # Initialize the model
    if args.seg_model_name.lower() == 'nestedsharedatrousresunet':
        model = NestedSharedAtrousResUNet(num_classes=args.num_classes, num_channels=args.num_channels, width=args.width)
    else:
        print('Model not found')
        exit()

    # Load the model state dictionary from the specified file if the --state argument is provided
    if args.state:
        try:
            # Try to load the model's state dictionary
            if args.cuda:
                # Load the model's state dictionary onto GPU if CUDA is available
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cuda')))
            else:
                # Load the model's state dictionary onto CPU if CUDA is not available
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))

            # Print a message indicating that the model state is loaded successfully
            print("Model state loaded")

        except AssertionError:
            # Handle the case where loading the model state fails due to an AssertionError
            print("Assertion error occurred while loading model state")

            # Attempt to load the model's state dictionary with default behavior
            model.load_state_dict(torch.load(args.state, map_location=lambda storage, loc: storage))

    # Move model to GPU if available
    if args.cuda:
        # Check if multiple GPUs are available and multi-GPU training is enabled
        if torch.cuda.device_count() > 1 and args.multi_gpu:
            print("Multi-GPU training enabled")
            # If so, wrap the model with DataParallel to utilize multiple GPUs
            model = torch.nn.DataParallel(model)
        else:
            # If only one GPU is available or multi-GPU training is not enabled
            if args.gpu is not None:
                # Set the specified GPU device if provided
                torch.cuda.set_device(args.gpu)
                
        # Move the model to the GPU
        model = model.cuda()

        # Optionally enable CuDNN optimization
        if args.cudnn:
            print('Using CUDNN')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    # Perform visualization or training or evaluation based on mode
    if args.mode == 'vis':
        plot_augmented_train_data(args)
    if args.mode == 'train':
        train(args, model)
    if args.mode == 'eval':
        evaluate(args, model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Hardware configuration arguments
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--gpu', type=int, help='Specify GPU device ID')
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training')
    parser.add_argument('--cudnn', action='store_true', help='Enable cuDNN acceleration')

    # Dataset and model directories
    parser.add_argument('--train_image_dir_mateusz', type=str, default='./data/SegNetWarm-Mateusz-coarse/all-images/')
    parser.add_argument('--train_mask_dir_mateusz', type=str, default='./data/SegNetWarm-Mateusz-coarse/all-masks/')
    parser.add_argument('--train_image_dir_openeds', type=str, default='./data/OpenEDS-coarse/all-images/')
    parser.add_argument('--train_mask_dir_openeds', type=str, default='./data/OpenEDS-coarse/all-masks/')
    parser.add_argument('--test_image_dir_newborn', type=str, default='./data/Piotr-NB-Dataset/all-images/')
    parser.add_argument('--test_mask_dir_newborn', type=str, default='./data/Piotr-NB-Dataset/all-masks/')
    parser.add_argument('--circle_model_path', type=str, default='./models/convnext_tiny-1076-0.030622-maskIoU-0.938355.pth')

    # Model hyperparameters
    parser.add_argument('--seg_model_name', type=str, default='nestedsharedatrousresunet', help='Segmentation model name')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of segmentation classes')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--width', type=int, default=32, help='Width of the model architecture')
    parser.add_argument('--circle_model_name', type=str, default='convnext', help='Name of the circle detection model')
    parser.add_argument('--state', type=str, help='Path to the model state file.')

    # Training hyperparameters
    parser.add_argument('--mode', default="vis", help='Mode of operation (train/eval/vis)')
    parser.add_argument('--aug_num_repetitions', type=int, default=2, help='Number of augmentations per input data')
    parser.add_argument('--pupil_pixel_range', type=tuple, default=(109, 200), help='Range of pixel values for pupil color adjustment')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--log_batch', type=int, default=10, help='See training performance 10 batch after after')
    parser.add_argument('--loss_type', type=str, default='cross_entropy+dice', help='Type of loss function')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--solver_name', type=str, default='madgrad', help='Name of optimizer (madgrad/Adam/SGD)')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--log_txt', action='store_true', help='Enable logging of training output to a text file.')
    parser.add_argument('--tag', type=str, default='coarsemasks')
    parser.add_argument('--label_smoothing', action='store_true')

    args = parser.parse_args()

    main(args)