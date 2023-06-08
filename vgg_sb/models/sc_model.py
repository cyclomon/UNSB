import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import losses
import numpy as np

class SCModel(BaseModel):
    """
    This class implements the unpaired image translation model with spatially correlative loss
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        :param parser: original options parser
        :return: the modified parser
        """
        parser.set_defaults(no_dropout=True)

        parser.add_argument('--attn_layers', type=str, default='4, 7, 9', help='compute spatial loss on which layers')
        parser.add_argument('--patch_nums', type=float, default=256, help='select how many patches for shape consistency, -1 use all')
        parser.add_argument('--patch_size', type=int, default=64, help='patch size to calculate the attention')
        parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')
        parser.add_argument('--use_norm', action='store_true', help='normalize the feature map for FLSeSim')
        parser.add_argument('--learned_attn', action='store_true', help='use the learnable attention map')
        parser.add_argument('--augment', action='store_true', help='use data augmentation for contrastive learning')
        parser.add_argument('--T', type=float, default=0.07, help='temperature for similarity')
        parser.add_argument('--lambda_spatial', type=float, default=10.0, help='weight for spatially-correlative loss')
        parser.add_argument('--lambda_spatial_idt', type=float, default=0.0, help='weight for idt spatial loss')
        parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='weight for feature consistency loss')
        parser.add_argument('--lambda_style', type=float, default=0.0, help='weight for style loss')
        parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping')
        parser.add_argument('--lambda_gradient', type=float, default=0.0, help='weight for the gradient penalty')
        parser.add_argument('--lmda', type=float, default=0.9, help='temperature for NCE loss')
        return parser

    def __init__(self, opt):
        """
        Initialize the translation losses
        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out
        self.loss_names = ['style', 'G_s', 'per', 'D_real', 'D_fake', 'G_GAN','ENT']
        # specify the images you want to save/display
        self.visual_names = ['real_A', 'fake_B' , 'real_B']
        
        if self.opt.phase == 'test':
            self.visual_names = ['real']
            for NFE in range(self.opt.num_timesteps):
                fake_name = 'fake_' + str(NFE+1)
                self.visual_names.append(fake_name)
        
        # if self.opt.phase == 'test':
        #     self.visual_names = ['real', 'fake_1','fake_2','fake_3','fake_4','fake_5']
        # specify the models you want to save to the disk
        self.model_names = ['G', 'D'] if self.isTrain else ['G']
        # define the networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
                                opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        # self.netS = networks.define_G(opt.input_nc*2, opt.output_nc*2, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
        #                         opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        # define the training process
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netE = networks.define_D(opt.output_nc*4, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
            if opt.lambda_identity > 0.0 or opt.lambda_spatial_idt > 0.0:
                # only works when input and output images have the same number of channels
                self.visual_names.append('idt_B')
                if opt.lambda_identity > 0.0:
                    self.loss_names.append('idt_B')
                if opt.lambda_spatial_idt > 0.0:
                    self.loss_names.append('G_s_idt_B')
                assert (opt.input_nc == opt.output_nc)
            if opt.lambda_gradient > 0.0:
                self.loss_names.append('D_Gradient')
            self.fake_B_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            # define the loss function
            self.netPre = losses.VGG16().to(self.device)
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionStyle = losses.StyleLoss().to(self.device)
            self.criterionFeature = losses.PerceptualLoss().to(self.device)
            self.criterionSpatial = losses.SpatialCorrelativeLoss(opt.loss_mode, opt.patch_nums, opt.patch_size, opt.use_norm,
                                    opt.learned_attn, gpu_ids=self.gpu_ids, T=opt.T).to(self.device)
            self.normalization = losses.Normalization(self.device)
            # define the contrastive loss
            if opt.learned_attn:
                self.netF = self.criterionSpatial
                self.model_names.append('F')
                self.loss_names.append('spatial')
            else:
                self.set_requires_grad([self.netPre], False)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_E = torch.optim.Adam(itertools.chain(self.netE.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The learnable spatially-correlative map is defined in terms of the shape of the intermediate, extracted features
        of a given network (encoder or pretrained VGG16). Because of this, the weights of spatial are initialized at the
        first feedforward pass with some input images
        :return:
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.isTrain:
            self.backward_G()
            self.optimizer_G.zero_grad()
            if self.opt.learned_attn:
                self.optimizer_F = torch.optim.Adam([{'params': list(filter(lambda p:p.requires_grad, self.netPre.parameters())), 'lr': self.opt.lr*0.0},
                                        {'params': list(filter(lambda p:p.requires_grad, self.netF.parameters()))}],
                                         lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
                self.optimizer_F.zero_grad()

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps
        :param input: include the data itself and its metadata information
        :return:
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A2 = input['A2' if AtoB else 'B2'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.opt.isTrain and self.opt.augment:
            self.aug_A = input['A_aug' if AtoB else 'B_aug'].to(self.device)
            self.aug_A2 = input['A_aug2' if AtoB else 'B_aug2'].to(self.device)
            self.aug_B = input['B_aug' if AtoB else 'A_aug'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1),times])
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs =  self.real_A.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()).long()
        self.time_idx = time_idx
        self.timestep     = times[time_idx]
        
        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item()+1):
                # print(t)
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_1,_     = self.netG(Xt, time_idx, z)
                
                Xt2       = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
                time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time     = times[time_idx]
                z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                Xt_12,_     = self.netG(Xt2, time_idx, z)
                
                
                if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0):
                    XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1B,_ = self.netG(XtB, time_idx, z)
            if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0):
                self.XtB = XtB.detach()
            self.Xt = Xt.detach()
            self.Xt2 = Xt2.detach()
                # self.Xt2 = Xt2
                
        
        z_in    = torch.randn(size=[2*bs,4*self.opt.ngf]).to(self.real_A.device)
        z_in2    = torch.randn(size=[bs,4*self.opt.ngf]).to(self.real_A.device)
        """Run forward pass"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0) and self.opt.isTrain else self.real_A
        
        self.realt = torch.cat((self.Xt, self.XtB), dim=0) if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0) and self.opt.isTrain else self.Xt
        
        self.fake, _ = self.netG(self.realt,self.time_idx,z_in)
        self.fake_B2, _ =  self.netG(self.Xt2,self.time_idx,z_in2)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0) and self.opt.isTrain:
            self.idt_B = self.fake[self.real_A.size(0):]
        if self.opt.phase == 'test':
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1/(i+1) for i in range(T-1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1),times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs =  self.real.size(0)
            time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()).long()
            self.time_idx = time_idx
            self.timestep     = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    # print(t)
                    if t > 0:
                        delta = times[t] - times[t-1]
                        denom = times[-1] - times[t-1]
                        inter = (delta / denom).reshape(-1,1,1,1)
                        scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                    Xt       = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                    time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                    time     = times[time_idx]
                    z        = torch.randn(size=[self.real_A.shape[0],4*self.opt.ngf]).to(self.real_A.device)
                    Xt_1,_     = self.netG(Xt, time_idx, z)
                    setattr(self, "fake_"+str(t+1), Xt_1)
                    # visuals.append(Xt_1)
                    
#             self.fake_1 = visuals[0]
#             self.fake_2 = visuals[1]

#             self.fake_3 = visuals[2]
#             self.fake_4 = visuals[3]
#             self.fake_5 = visuals[4]
    def backward_F(self):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        norm_real_A, norm_real_B, norm_fake_B = self.normalization((self.real_A + 1) * 0.5), self.normalization((self.real_B + 1) * 0.5), self.normalization((self.fake_B.detach() + 1) * 0.5)
        if self.opt.augment:
            norm_aug_A, norm_aug_B = self.normalization((self.aug_A + 1) * 0.5), self.normalization((self.aug_B + 1) * 0.5)
            norm_real_A = torch.cat([norm_real_A, norm_real_A], dim=0)
            norm_fake_B = torch.cat([norm_fake_B, norm_aug_A], dim=0)
            norm_real_B = torch.cat([norm_real_B, norm_aug_B], dim=0)
        self.loss_spatial = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, norm_real_B)

        self.loss_spatial.backward()
    def backward_E(self):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        bs =  self.real_A.size(0)
        # z_in    = torch.randn(size=[2*bs,self.opt.style_dim]).to(self.real_A.device)
        # z_in2    = torch.randn(size=[bs,self.opt.style_dim]).to(self.real_A.device)
        # ts2 = torch.cat([self.timestep]*2,dim=0)
        # self.fake = self.netG(self.realt,ts2,z_in)
        # self.fake_B2 = self.netG(self.Xt2,self.timestep,z_in2)
        # self.fake_B = self.fake[:self.real_A.size(0)]
        
        """Calculate GAN loss for the discriminator"""
        
        XtXt_1 = torch.cat([self.Xt,self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.Xt2,self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        self.loss_E = -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2
        # opt_E.zero_grad()
        self.loss_E.backward()
        # opt_E.step()
        
        # fake = self.fake_B.detach()
        # XtXt_1 = torch.cat([self.Xt,fake], dim=1)
        # noise = torch.randn_like(XtXt_1).to(fake.device)
        # XtXt_1_noise = XtXt_1.detach() + 0.1 * noise
        # # XtXt_2 = torch.cat([self.Xt2,fake], dim=1)
        # z = torch.zeros(size=[bs,4*self.opt.ngf]).to(fake.device)
        # out,_ = self.netS(XtXt_1_noise, self.time_idx, z)
        # self.loss_S = (noise + 0.1 * out).square().mean()

        # self.loss_S.backward()
    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator
        :param netD: the discriminator D
        :param real: real images
        :param fake: images generated by a generator
        :return: discriminator loss
        """
        # real
        real.requires_grad_()
        pred_real = netD(real,self.time_idx)
        self.loss_D_real = self.criterionGAN(pred_real, True, is_dis=True)
        # fake
        pred_fake = netD(fake,self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_dis=True)
        # combined loss
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        # gradient penalty
        if self.opt.lambda_gradient > 0.0:
            self.loss_D_Gradient, _ = losses.cal_gradient_penalty(netD, real, fake, real.device, lambda_gp=self.opt.lambda_gradient)#
            loss_D += self.loss_D_Gradient
        loss_D.backward()

        return loss_D

    def backward_D(self):
        """Calculate the GAN loss for discriminator"""
        # fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_B, fake_B.detach())

    def backward_G(self):
        """Calculate the loss for generator G_A"""
        l_style = self.opt.lambda_style
        l_per = self.opt.lambda_perceptual
        l_sptial = self.opt.lambda_spatial
        l_idt = self.opt.lambda_identity
        l_spatial_idt = self.opt.lambda_spatial_idt
        # GAN loss
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B,self.time_idx), True)
        # different structural loss
        norm_real_A = self.normalization((self.real_A + 1) * 0.5)
        norm_fake_B = self.normalization((self.fake_B + 1) * 0.5)
        norm_real_B = self.normalization((self.real_B + 1) * 0.5)
        self.loss_style = self.criterionStyle(norm_real_B, norm_fake_B) * l_style if l_style > 0 else 0
        self.loss_per = self.criterionFeature(norm_real_A, norm_fake_B) * l_per if l_per > 0 else 0
        self.loss_G_s = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, None) * l_sptial if l_sptial > 0 else 0
        # identity loss
        if l_spatial_idt > 0:
            norm_fake_idt_B = self.normalization((self.idt_B + 1) * 0.5)
            self.loss_G_s_idt_B = self.Spatial_Loss(self.netPre, norm_real_B, norm_fake_idt_B, None) * l_spatial_idt
        else:
            self.loss_G_s_idt_B = 0
        self.loss_idt_B = self.criterionIdt(self.real_B, self.idt_B) * l_idt if l_idt > 0 else 0
        loss_OT = self.loss_style + self.loss_per + self.loss_G_s + self.loss_G_s_idt_B + self.loss_idt_B
        XtXt_1 = torch.cat([self.Xt, self.fake_B], dim=1)
        XtXt_2 = torch.cat([self.Xt2, self.fake_B2], dim=1)
        # params   = list(self.netG.parameters() )
        bs = self.opt.batch_size
        
        # XtXt_1   = torch.cat([Xt,Xt_1], dim=1)
        ET_XY    = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
        self.loss_ENT = -(self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*ET_XY
        self.loss_ENT += self.opt.tau*torch.mean((self.Xt-self.fake_B)**2)
        self.loss_G = self.loss_style + self.loss_per + self.loss_G_s + self.loss_G_s_idt_B + self.loss_idt_B + self.loss_G_GAN + self.loss_ENT
        
        
#         grad_ENT,_ = self.netS(XtXt_1, self.time_idx, torch.zeros(size=[bs,4*self.opt.ngf]).to(self.real_A.device))
#         grad_ENT = grad_ENT[:,3:].detach() /bs
#         grad_ENT = torch.autograd.grad(outputs=self.fake_B, inputs=params, grad_outputs=grad_ENT, allow_unused=True, retain_graph=True)
#         grad_OT  = torch.autograd.grad(outputs=loss_OT, inputs=params, grad_outputs=torch.ones_like(loss_OT), allow_unused=True, retain_graph=True)
#         grad_ADV = torch.autograd.grad(outputs=self.loss_G_GAN, inputs=params, grad_outputs=torch.ones_like(self.loss_G_GAN), allow_unused=True)
#         for k,p in enumerate(params):
#             if grad_ENT[k] is not None:
                
#                 grad_SB = (self.opt.num_timesteps-self.time_idx[0])/self.opt.num_timesteps*self.opt.tau*grad_ENT[k] + grad_OT[k]
#                 grad_SB_norm = torch.norm(grad_SB)
#                 grad_ADV_norm = torch.norm(grad_ADV[k])
#                 factor = torch.minimum(grad_ADV_norm, grad_SB_norm) / (grad_SB_norm + 1e-10)
                
#                 p.grad = self.opt.lmda * grad_SB + grad_ADV[k]
        # self.loss_G = self.loss_G_GAN + self.loss_style + self.loss_per + self.loss_G_s + self.loss_G_s_idt_B + self.loss_idt_B
        
        # self.loss_G = self.loss_G_GAN + self.loss_style + self.loss_per + self.loss_G_s + self.loss_G_s_idt_B + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # forward
        self.forward()
        if self.opt.learned_attn:
            self.set_requires_grad([self.netF, self.netPre], True)
            self.optimizer_F.zero_grad()
            self.backward_F()
            self.optimizer_F.step()
        # D_A
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.set_requires_grad([self.netE], True)
        self.optimizer_E.zero_grad()
        self.backward_E()
        self.optimizer_E.step()
        # G_A
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netE], False)
        self.optimizer_G.zero_grad()
        if self.opt.learned_attn:
            self.set_requires_grad([self.netF, self.netPre], False)
        self.backward_G()
        self.optimizer_G.step()

    def Spatial_Loss(self, net, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers