import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from pix2pixHD import VGGLoss, MultiscaleDiscriminator, GANLoss
from models.ResnetSTNGenerator import ResnetSTNGenerator
from torch import nn


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_A', 'D_A', 'cycle_A', 'G_B', 'D_B', 'cycle_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.n_blocks = 1

        self.netG_A = ResnetSTNGenerator(opt.input_nc, opt.output_nc, ngf=opt.ngf, norm_layer=nn.InstanceNorm2d, n_blocks=1, stn_mode='truth').cuda()
        self.netG_B = ResnetSTNGenerator(opt.input_nc, opt.output_nc, ngf=opt.ngf, norm_layer=nn.InstanceNorm2d, n_blocks=1, stn_mode='truth').cuda()

        if self.isTrain:  # define discriminators
            # self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_A = MultiscaleDiscriminator(opt.output_nc, n_layers=4, ndf=opt.ndf, num_D=3, use_sigmoid=False, getIntermFeat=True).cuda()
            self.netD_B = MultiscaleDiscriminator(opt.output_nc, n_layers=4, ndf=opt.ndf, num_D=3, use_sigmoid=False, getIntermFeat=True).cuda()

        self.eyetrans = torch.cat(6*[torch.eye(3)[None,:]]).cuda()
        self.eyetrans = self.eyetrans[None,:]

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            # self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionGAN = GANLoss(use_lsgan=True).cuda()
            self.criterionCycle = torch.nn.L1Loss().cuda()
            self.criterionIdt = torch.nn.L1Loss().cuda()
            self.fm_loss = torch.nn.L1Loss().cuda()
            self.vgg_loss = VGGLoss().cuda()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # def set_input(self, real_A, fortrans, fortran_imgs, invtrans, invtran_imgs, real_B):
    def set_input(self, real_A, real_B):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_A = real_A.to(self.device)
        # self.fortrans = fortrans.to(self.device)
        # self.fortran_imgs = fortran_imgs.to(self.device)
        # self.invtrans = invtrans.to(self.device)
        # self.invtran_imgs = invtran_imgs.to(self.device)
        self.real_B = real_B.to(self.device)
        # self.real_B = self.real_A

        # self.target = self.fortran_imgs[:,0]
        
        # Feature matching target
        self.target = self.real_A

    def forward(self, show=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_B = self.netG_A(self.real_A, self.eyetrans, cycle=True)
        self.rec_A  = self.netG_B(self.fake_B, self.eyetrans, cycle=True)
        self.fake_A = self.netG_B(self.real_B, self.eyetrans, cycle=True)
        self.rec_B  = self.netG_A(self.fake_A, self.eyetrans, cycle=True)

        # self.fake_B = self.netG_A(self.real_A, self.fortrans)
        # self.rec_A  = self.netG_B(self.fake_B, self.invtrans)
        # self.fake_A = self.netG_B(self.real_B, self.invtrans)
        # self.rec_B  = self.netG_A(self.fake_A, self.fortrans)

        # self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        if show:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 4)

            combo = torch.ones_like(self.real_A)
            combo[self.target != -1.0] = self.fake_B[self.target != -1.0]
            combo[self.target == -1.0] = self.fake_A[self.target == -1.0]

            # Truth
            ax[0][0].imshow(self.real_A[0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)
            ax[0][1].imshow(combo[0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)
            # ax[0][2].imshow(self.invtran_imgs[0,0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)
            ax[0][3].imshow(self.real_B[0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)

            # Gen
            ax[1][0].imshow(self.fake_B[0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)
            ax[1][0].set_title('fake_B')
            ax[1][1].imshow(self.rec_A[0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)
            ax[1][1].set_title('rec_A')
            ax[1][2].imshow(self.fake_A[0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)
            ax[1][2].set_title('fake_A')
            ax[1][3].imshow(self.rec_B[0,:].permute(1,2,0).detach().cpu(), cmap='gray', vmin=-1, vmax=1)
            ax[1][3].set_title('rec_B')
            plt.tight_layout()
            plt.show()
            plt.close()
            # quit()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward(retain_graph=True)
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.target, fake_A)

    def boosted_loss(self, netD, output, target, lambda_fm=5, lambda_vgg=2):

        pred_real = netD(target)
        pred_fake = netD(output)

        loss_G_GAN_Feat = 0
        for i in range(3):
            for j in range(4):
                weight = 1.0 / (2**((4-1)-i))
                loss_G_GAN_Feat += weight * self.fm_loss(pred_fake[i][j], pred_real[i][j].detach())
        loss_G_GAN_Feat = loss_G_GAN_Feat * lambda_fm
        
        loss_G_VGG = self.vgg_loss(output, target) * lambda_vgg
    
        return loss_G_GAN_Feat + loss_G_VGG

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = 10
        lambda_B = 10

        feats_fake_A = -(torch.ones_like(self.target))
        feats_fake_B = -(torch.ones_like(self.target))
        feats_rec_A = -(torch.ones_like(self.target))
        feats_rec_B = -(torch.ones_like(self.target))

        feats_fake_B[self.target != -1.0] = self.fake_B[self.target != -1.0]
        feats_fake_A[self.target != -1.0] = self.fake_A[self.target != -1.0]
        feats_rec_B[self.target != -1.0] = self.rec_B[self.target != -1.0]
        feats_rec_A[self.target != -1.0] = self.rec_A[self.target != -1.0]
        
        # # Feature loss
        # feat_loss_A = self.boosted_loss(self.netD_A, feats_fake_B, self.target)
        feat_loss_A = self.boosted_loss(self.netD_A, feats_fake_B, self.target) + self.boosted_loss(self.netD_A, feats_rec_B, self.target)
        feat_loss_B = self.boosted_loss(self.netD_B, feats_fake_A, self.target) + self.boosted_loss(self.netD_B, feats_rec_A, self.target)
        # feat_loss_B = self.boosted_loss(self.netD_B, feats_fake_A, self.target)

        # feat_loss_A = self.vgg_loss(feats_fake_B, self.target)
        # feat_loss_B = self.vgg_loss(feats_fake_A, self.target)

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) + (feat_loss_A*5)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) + (feat_loss_B*5)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.target) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward(retain_graph=True)

    def optimize_parameters(self, show=False):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(show)      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()   # set G_A and G_B's gradients to zero
        self.backward_G() # calculate gradients for G_A and G_B
        self.optimizer_G.step()        # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
