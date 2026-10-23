from templates import *


def grade224_autoenc_cls():
    conf = grade224_autoenc_130M()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{grade224_autoenc_130M().name}/latent.pkl'
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{grade224_autoenc_130M().name}/last.ckpt',
    )
    conf.name = 'grade224_autoenc_cls'
    return conf


def grade256_autoenc_cls():
    '''We first train the encoder on grade dataset then use it as a pretrained to train a linear classifer on CelebA dataset with attribute labels'''
    conf = grade256_autoenc()
    conf.train_mode = TrainMode.manipulate
    conf.manipulate_mode = ManipulateMode.celebahq_all
    conf.manipulate_znormalize = True
    conf.latent_infer_path = f'checkpoints/{grade256_autoenc().name}/latent.pkl'  # we train on Celeb dataset, not grade
    conf.batch_size = 32
    conf.lr = 1e-3
    conf.total_samples = 300_000
    # use the pretraining trick instead of contiuning trick
    conf.pretrain = PretrainConfig(
        '130M',
        f'checkpoints/{grade256_autoenc().name}/last.ckpt',
    )
    conf.name = 'grade256_autoenc_cls'
    return conf
