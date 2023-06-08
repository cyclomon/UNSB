from .sc_model import SCModel


class SinSCModel(SCModel):
    """
    This class implements the single image translation
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        :param parser: original options parser
        :param is_train: whether training phase or test phase. You can use this flag to add training-specific or test-specific options
        :return: the modified parser
        """
        parser = SCModel.modify_commandline_options(parser,  is_train)

        parser.set_defaults(
            dataset_mode='singleimage',
            netG='stylegan2',
            stylegan2_G_num_downsampling=2,
            netD="stylegan2",
            gan_mode="nonsaturating",
            num_patches=1,
            attn_layers="4,7,9",
            lambda_spatial=10.0,
            lambda_identity=0.0,
            lambda_gradient=1.0,
            lambda_spatial_idt=0.0,
            ngf=8,
            ndf=8,
            lr=0.001,
            beta1=0.0,
            beta2=0.99,
            load_size=1024,
            crop_size=128,
            preprocess="zoom_and_patch",
            D_patch_size=None,
        )

        if is_train:
            parser.set_defaults(preprocess="zoom_and_patch",
                                batch_size=16,
                                save_epoch_freq=1,
                                save_latest_freq=20000,
                                n_epochs=4,
                                n_epochs_decay=4,
                                )
        else:
            parser.set_defaults(preprocess="none",  # load the whole image as it is
                                batch_size=1,
                                num_test=1,
                                )

        return parser

    def __init__(self, opt):
        super().__init__(opt)

