"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.datasets import MultipleEMGDatasetFirst, MultipleEMGDatasetSecond, MultipleEMGDatasetThird, OmniglotDataset, MiniImageNet, EMGDataset
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from few_shot.models import FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH


class Args(object):
    def __init__(self, dataset, n, k, q):
        self.dataset = dataset
        self.n = n
        self.k = k
        self.q = q
        self.order = 1
        self.inner_train_steps = 1
        self.inner_val_steps = 3
        self.inner_lr = 0.4
        self.meta_lr = 0.001
        self.epochs = 20
        self.epoch_len = 100
        self.eval_batches = 20
        self.meta_batch_size = 16


def run(args):
    if args.dataset == 'omniglot':
        dataset_class = OmniglotDataset
        fc_layer_size = 64
        num_input_channels = 1
    elif args.dataset == 'miniImageNet':
        dataset_class = MiniImageNet
        fc_layer_size = 1600
        num_input_channels = 3
    elif args.dataset == 'emg':
        dataset_class = EMGDataset
        fc_layer_size = 64
        num_input_channels = 8
    elif args.dataset == 'memg1':
        dataset_class = MultipleEMGDatasetFirst
        fc_layer_size = 64
        # num_input_channels = 8
        num_input_channels = 16
    elif args.dataset == 'memg2':
        dataset_class = MultipleEMGDatasetSecond
        fc_layer_size = 64
        # num_input_channels = 8
        num_input_channels = 16
    elif args.dataset == 'memg3':
        dataset_class = MultipleEMGDatasetThird
        fc_layer_size = 64
        # num_input_channels = 8
        num_input_channels = 16

    else:
        raise(ValueError('Unsupported dataset'))

    param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
                f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'
    print(param_str)

    ###################
    # Create datasets #
    ###################
    background = dataset_class('background')
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                       num_tasks=args.meta_batch_size),
        num_workers=1
    )
    evaluation = dataset_class('evaluation')
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                       num_tasks=args.meta_batch_size),
        num_workers=1
    )

    ############
    # Training #
    ############
    print(f'Training MAML on {args.dataset}...')
    meta_model = FewShotClassifier(
        num_input_channels, args.k, fc_layer_size).to(device, dtype=torch.double)
    meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    loss_fn = nn.CrossEntropyLoss().to(device)

    def prepare_meta_batch(n, k, q, meta_batch_size):
        def prepare_meta_batch_(batch):
            x, y = batch
            # Reshape to `meta_batch_size` number of tasks. Each task contains
            # n*k support samples to train the fast model on and q*k query samples to
            # evaluate the fast model on and generate meta-gradients

            x = x.reshape(meta_batch_size, n*k + q*k,
                          num_input_channels, x.shape[-2], x.shape[-1])

            # Move to device
            x = x.double().to(device)
            # Create label
            y = create_nshot_task_label(k, q).repeat(meta_batch_size)
            return x, y

        return prepare_meta_batch_

    callbacks = [
        EvaluateFewShot(
            eval_fn=meta_gradient_step,
            num_tasks=args.eval_batches,
            n_shot=args.n,
            k_way=args.k,
            q_queries=args.q,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_meta_batch(
                args.n, args.k, args.q, args.meta_batch_size),
            # MAML kwargs
            inner_train_steps=args.inner_val_steps,
            inner_lr=args.inner_lr,
            device=device,
            order=args.order,
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/maml/{param_str}.pth',
            monitor=f'val_{args.n}-shot_{args.k}-way_acc'
        ),
        ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
        CSVLogger(PATH + f'/logs/maml/{param_str}.csv'),
    ]

    fit(
        meta_model,
        meta_optimiser,
        loss_fn,
        epochs=args.epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_meta_batch(
            args.n, args.k, args.q, args.meta_batch_size),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=meta_gradient_step,
        fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                             'train': True,
                             'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                             'inner_lr': args.inner_lr},
    )


if __name__ == '__main__':
    setup_dirs()
    # assert torch.cuda.is_available()
    device = torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    ##############
    # Parameters #
    ##############
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='emg')
    # parser.add_argument('--dataset', default='omniglot')
    # parser.add_argument('--n', default=7, type=int)
    # parser.add_argument('--k', default=26, type=int)
    # Number of examples per class to calculate meta gradients with
    # parser.add_argument('--q', default=1, type=int)
    # parser.add_argument('--inner-train-steps', default=1, type=int)
    # parser.add_argument('--inner-val-steps', default=3, type=int)
    # parser.add_argument('--inner-lr', default=0.4, type=float)
    # parser.add_argument('--meta-lr', default=0.001, type=float)
    # parser.add_argument('--meta-batch-size', default=16, type=int)
    # parser.add_argument('--order', default=1, type=int)
    # parser.add_argument('--epochs', default=20, type=int)
    # parser.add_argument('--epoch-len', default=100, type=int)
    # parser.add_argument('--eval-batches', default=20, type=int)

    # args = parser.parse_args()
    datasets = ['memg1', 'memg2', 'memg3']
    params = [
        [3, 5, 1],
        [3, 5, 5],
        [3, 5, 7],
        [3, 3, 7],
        [3, 2, 7],
        [7, 3, 5],
        [26, 5, 7],
    ]
    for d in datasets:
        for p in params:
            args = Args(d, p[0], p[1], p[2])
            run(args)
