from torch.optim import lr_scheduler
from . import cyclegan_networks, stylegan_networks


##################################################################################
# Networks
##################################################################################
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    """
    Create a generator
    :param input_nc: the number of channels in input images
    :param output_nc: the number of channels in output images
    :param ngf: the number of filters in the first conv layer
    :param netG: the architecture's name: resnet_9blocks | munit | stylegan2
    :param norm: the name of normalization layers used in the network: batch | instance | none
    :param use_dropout: if use dropout layers.
    :param init_type: the name of our initialization method.
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :param no_antialias: use learned down sampling layer or not
    :param no_antialias_up: use learned up sampling layer or not
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2
    :param opt: options
    :return:
    """
    norm_value = cyclegan_networks.get_norm_layer(norm)

    if netG == 'resnet_9blocks':
        net = cyclegan_networks.ResnetGenerator(input_nc, output_nc, ngf, norm_value, use_dropout, n_blocks=9, no_antialias=no_antialias, no_antialias_up=no_antialias_up, opt=opt)
    elif netG == 'stylegan2':
        net = stylegan_networks.StyleGAN2Generator(input_nc, output_nc, ngf, opt=opt)
    elif netG == 'resnet_9blocks_cond':
        net = cyclegan_networks.ResnetGenerator_ncsn(input_nc, output_nc, ngf, norm_value, use_dropout, n_blocks=9, no_antialias=no_antialias, no_antialias_up=no_antialias_up, opt=opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return cyclegan_networks.init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    """
    Create a discriminator
    :param input_nc: the number of channels in input images
    :param ndf: the number of filters in the first conv layer
    :param netD: the architecture's name
    :param n_layers_D: the number of conv layers in the discriminator; effective when netD=='n_layers'
    :param norm: the type of normalization layers used in the network
    :param init_type: the name of the initialization method
    :param init_gain: scaling factor for normal, xavier and orthogonal
    :param no_antialias: use learned down sampling layer or not
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2
    :param opt: options
    :return:
    """
    norm_value = cyclegan_networks.get_norm_layer(norm)
    if netD == 'basic':
        net = cyclegan_networks.NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_value, no_antialias)
    elif netD == 'bimulti':
        net = cyclegan_networks.D_NLayersMulti(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_value, num_D=2)
    elif netD == 'basic_cond':
        net = cyclegan_networks.NLayerDiscriminator_ncsn(input_nc, ndf, n_layers_D, norm_value, no_antialias)
    elif 'stylegan2' in netD:
        net = stylegan_networks.StyleGAN2Discriminator(input_nc, ndf, opt=opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return cyclegan_networks.init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netD))


###############################################################################
# Helper Functions
###############################################################################
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler