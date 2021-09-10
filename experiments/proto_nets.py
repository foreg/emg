"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from typing import Any
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from few_shot.datasets import MultipleEMGDatasetFirst, MultipleEMGDatasetSecond, MultipleEMGDatasetThird, MultipleEMGDatasetAll, OmniglotDataset, MiniImageNet, EMGDataset
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH


class Args(object):
    def __init__(self, dataset, distance, n_train, n_test, k_train, k_test, q_train, q_test):
        self.dataset = dataset
        self.distance = distance
        self.n_train = n_train
        self.n_test = n_test
        self.k_train = k_train
        self.k_test = k_test
        self.q_train = q_train
        self.q_test = q_test


def run(args):
    evaluation_episodes = 1000
    episodes_per_epoch = 100

    if args.dataset == 'omniglot':
        n_epochs = 40
        dataset_class = OmniglotDataset
        num_input_channels = 1
        drop_lr_every = 20
    elif args.dataset == 'emg':
        n_epochs = 10
        dataset_class = EMGDataset
        num_input_channels = 8
        drop_lr_every = 20
    elif args.dataset == 'memg1':
        n_epochs = 40
        dataset_class = MultipleEMGDatasetFirst
        # num_input_channels = 8
        num_input_channels = 16
        drop_lr_every = 20
    elif args.dataset == 'memg2':
        n_epochs = 40
        dataset_class = MultipleEMGDatasetSecond
        # num_input_channels = 8
        num_input_channels = 16
        drop_lr_every = 20
    elif args.dataset == 'memg3':
        n_epochs = 40
        dataset_class = MultipleEMGDatasetThird
        # num_input_channels = 8
        num_input_channels = 16
        drop_lr_every = 20
    elif args.dataset == 'memg123':
        n_epochs = 40
        dataset_class = MultipleEMGDatasetAll
        num_input_channels = 24
        drop_lr_every = 20
    elif args.dataset == 'miniImageNet':
        n_epochs = 80
        dataset_class = MiniImageNet
        num_input_channels = 3
        drop_lr_every = 40
    else:
        raise(ValueError, 'Unsupported dataset')

    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    print(param_str)

    ###################
    # Create datasets #
    ###################
    background = dataset_class('background')
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(
            background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
        num_workers=4
    )
    evaluation = dataset_class('evaluation')
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(
            evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
        num_workers=4
    )

    #########
    # Model #
    #########
    model = get_few_shot_encoder(num_input_channels)
    model.to(device, dtype=torch.double)

    ############
    # Training #
    ############
    print(f'Training Prototypical network on {args.dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()

    def lr_schedule(epoch, lr):
        # Drop lr every 2000 episodes
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr

    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(
                args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}.pth',
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
    ]

    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(
            args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
    )


if __name__ == '__main__':
    setup_dirs()
    # assert torch.cuda.is_available()
    device = torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    # # parser.add_argument('--dataset', default='emg')
    # parser.add_argument('--dataset', default='memg3')
    # parser.add_argument('--distance', default='l2')
    # parser.add_argument('--n-train', default=5, type=int)
    # parser.add_argument('--n-test', default=1, type=int)
    # # parser.add_argument('--k-train', default=26, type=int)
    # parser.add_argument('--k-train', default=5, type=int)
    # # parser.add_argument('--k-test', default=13, type=int)
    # parser.add_argument('--k-test', default=3, type=int)
    # parser.add_argument('--q-train', default=5, type=int)
    # parser.add_argument('--q-test', default=1, type=int)
    # args = parser.parse_args()r
    datasets = ['memg1']
    # datasets = ['memg2', 'memg3']
    # params = [
    #     [1, 3, 5, 1, 2, 1],
    #     [1, 7, 5, 1, 5, 1],
    #     [1, 7, 1, 1, 1, 1],
    #     [1, 7, 10, 1, 7, 1],
    #     [3, 7, 1, 1, 3, 1],
    #     [3, 7, 5, 1, 3, 1],
    #     [3, 7, 5, 1, 7, 1],
    #     [3, 7, 10, 1, 7, 1],
    #     [5, 7, 5, 3, 7, 1],
    #     [5, 7, 10, 3, 7, 1],
    #     [5, 7, 10, 3, 7, 5],
    #     [5, 7, 10, 3, 7, 10],
    # ]
    params = [
        # [2, 2, 10, 2, 2, 1],
        # [3, 3, 10, 3, 3, 1],
        [5, 3, 10, 5, 3, 1],
        # [7, 3, 10, 7, 3, 1],
        # [3, 7, 10, 3, 3, 1],
        # [3, 7, 10, 3, 7, 1],
        # [3, 5, 10, 3, 3, 1],
        # [3, 5, 10, 3, 5, 1],
        # [1, 3, 10, 1, 3, 1],
        # [1, 2, 10, 1, 2, 1],
    ]
    for d in datasets:
        for p in params:
            args = Args(d, 'l2', p[0],
                        p[3], p[1], p[4], p[2], p[5])
            run(args)
