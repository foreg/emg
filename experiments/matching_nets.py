"""
Reproduce Matching Network results of Vinyals et al
"""
import argparse
from torch.nn import parameter
from torch.utils.data import DataLoader
from torch.optim import Adam

from few_shot.datasets import MultipleEMGDatasetFirst, MultipleEMGDatasetSecond, MultipleEMGDatasetThird, MultipleEMGDatasetAll, OmniglotDataset, MiniImageNet, EMGDataset
from few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot
from few_shot.matching import matching_net_episode
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
        self.fce = None
        self.lstm_layers = 1
        self.unrolling_steps = 2


def run(args):
    evaluation_episodes = 1000
    episodes_per_epoch = 100

    if args.dataset == 'omniglot':
        n_epochs = 100
        dataset_class = OmniglotDataset
        num_input_channels = 1
        lstm_input_size = 64
    elif args.dataset == 'emg':
        n_epochs = 20
        dataset_class = EMGDataset
        num_input_channels = 8
        lstm_input_size = 64
    elif args.dataset == 'memg1':
        n_epochs = 20
        dataset_class = MultipleEMGDatasetFirst
        # num_input_channels = 8
        num_input_channels = 16
        lstm_input_size = 64
    elif args.dataset == 'memg2':
        n_epochs = 20
        dataset_class = MultipleEMGDatasetSecond
        # num_input_channels = 8
        num_input_channels = 16
        lstm_input_size = 64
    elif args.dataset == 'memg3':
        n_epochs = 20
        dataset_class = MultipleEMGDatasetThird
        # num_input_channels = 8
        num_input_channels = 16
        lstm_input_size = 64
    elif args.dataset == 'memg123':
        n_epochs = 20
        dataset_class = MultipleEMGDatasetAll
        num_input_channels = 24
        lstm_input_size = 64
    elif args.dataset == 'miniImageNet':
        n_epochs = 200
        dataset_class = MiniImageNet
        num_input_channels = 3
        lstm_input_size = 1600
    else:
        raise(ValueError, 'Unsupported dataset')

    param_str = f'{args.dataset}_n={args.n_train}_k={args.k_train}_q={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}_'\
                f'dist={args.distance}_fce={args.fce}'

    #########
    # Model #
    #########
    from few_shot.models import MatchingNetwork
    model = MatchingNetwork(args.n_train, args.k_train, args.q_train, args.fce, num_input_channels,
                            lstm_layers=args.lstm_layers,
                            lstm_input_size=lstm_input_size,
                            unrolling_steps=args.unrolling_steps,
                            device=device)
    model.to(device, dtype=torch.double)

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

    ############
    # Training #
    ############
    print(f'Training Matching Network on {args.dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()

    callbacks = [
        EvaluateFewShot(
            eval_fn=matching_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(
                args.n_test, args.k_test, args.q_test),
            fce=args.fce,
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/matching_nets/{param_str}.pth',
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc',
            # monitor=f'val_loss',
        ),
        ReduceLROnPlateau(patience=20, factor=0.5,
                          monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'),
        CSVLogger(PATH + f'/logs/matching_nets/{param_str}.csv'),
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
        fit_function=matching_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'fce': args.fce, 'distance': args.distance}
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
    # parser.add_argument('--fce', type=lambda x: x.lower()
    #                     [0] == 't')  # Quick hack to extract boolean
    # parser.add_argument('--distance', default='cosine')
    # parser.add_argument('--n-train', default=3, type=int)
    # parser.add_argument('--n-test', default=3, type=int)
    # parser.add_argument('--k-train', default=3, type=int)
    # parser.add_argument('--k-test', default=3, type=int)
    # parser.add_argument('--q-train', default=15, type=int)
    # parser.add_argument('--q-test', default=1, type=int)
    # parser.add_argument('--lstm-layers', default=1, type=int)
    # parser.add_argument('--unrolling-steps', default=2, type=int)
    # args = parser.parse_args()
    # datasets = ['memg1', 'memg2', 'memg3']
    datasets = ['memg123']
    params = [
        [1, 1, 15, 1, 1, 1],
        [1, 7, 15, 3, 7, 1],
        [3, 7, 15, 3, 7, 1],
        [3, 3, 15, 3, 3, 1],
        [3, 5, 15, 3, 5, 1],

    ]
    for d in datasets:
        for p in params:
            args = Args(d, 'cosine', p[0],
                        p[3], p[1], p[4], p[2], p[5])
            run(args)
