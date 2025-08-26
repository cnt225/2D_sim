import os
from omegaconf import OmegaConf
import torch
import numpy as np

# from .ae import (
#     AE,
#     VAE,
#     IRVAE
# )

from .modules import (
    # FC_vec,
    # FC_image,
    # vf_FC_vec,
    # vf_FC_image,
    # cfw_FC_vec,
    # UNET4MAPP,
    # UNET24MAPP,
    # IsotropicGaussian,
    # ConvNet28,
    # DeConvNet28,
    # ConvNet32,
    # DeConvNet32,
    # ConvNet64,
    # DeConvNet64,
    # vf_FC_vec_onSphere, # added for sphere toy
    # vf_FC_vec_ImgToSphere, # added for image to sphere
    vf_FC_vec_grasp, # added for grasping
    vf_FC_vec_motion, # added for motion planning
)

# from .fm import FM
# from .gpp import GPP
# from .mapp import MAPP
# from .mepp import MEPP
# from .unet import UNetModel
# from .unet2 import UNet
# from .rfm_sphere import RFM_sphere # added for sphere toy
# from .rcfm_sphere import RCFM_sphere  # image(conditional) to sphere
from .grasp_rcfm import GraspRCFM
from .motion_rcfm import MotionRCFM
# from .vnn_pointnet import VNNResnetPointnet
# from .feature_net import TimeLatentFeatureEncoder
# from .energy_net import EnergyNet
# from .grasp_dif import GraspDiffusionFields
from .dgcnn import DGCNN # pc to latent_vec
# # from .grasp_graspnet import GraspVAE, GraspGAN
# from .vn_dgcnn import VN_DGCNN
# from .grasp_dif_only import GraspDiffusionFieldsONLY

def get_model(model_cfg, *args, **kwargs):
    name = model_cfg["arch"]
    model = _get_model_instance(name)
    model = model(**model_cfg, **kwargs)

    if 'checkpoint' in model_cfg:
        checkpoint = torch.load(model_cfg['checkpoint'], map_location='cpu')

        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])

    return model


def _get_model_instance(name):
    try:
        return {
            # 'fm': get_flow_matching_model,
            # 'gpp': get_gpp_model,
            # 'mapp': get_mapp_model,
            # 'mepp': get_mepp_model,
            # 'vae': get_ae,
            # 'irvae': get_ae,
            # 'rfm': get_riemannian_flow_matching_model,
            # 'rcfm': get_riemannian_conditional_flow_matching_model,
            # 'cnn32': get_cnn32_model,
            # 'grasp_dif': get_grasp_dif_model,
            'grasp_rcfm': get_grasp_rcfm_model,
            'motion_rcfm': get_motion_rcfm_model,
            # 'grasp_vae': get_grasp_vae_gan_model,
            # 'grasp_gan': get_grasp_vae_gan_model,
            # 'grasp_dif_only':get_grasp_dif_only_model,
        }[name]
    except:
        raise ("Model {} not available".format(name))


def get_net(**kwargs):
    # if kwargs["arch"] == "fc_vec":
    #     in_dim = kwargs['in_dim']
    #     out_dim = kwargs['out_dim']
    #     l_hidden = kwargs["l_hidden"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = FC_vec(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         l_hidden=l_hidden,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "fc_image":
    #     in_dim = kwargs['in_dim']
    #     out_dim = kwargs['out_dim']
    #     l_hidden = kwargs["l_hidden"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = FC_image(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         l_hidden=l_hidden,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "vf_fc_vec":
    #     in_dim = kwargs['in_dim']
    #     out_dim = kwargs['out_dim']
    #     l_hidden = kwargs["l_hidden"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = vf_FC_vec(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         l_hidden=l_hidden,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "vf_fc_vec_onSphere":
    #     in_dim = kwargs['in_dim']
    #     out_dim = kwargs['out_dim']
    #     l_hidden = kwargs["l_hidden"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = vf_FC_vec_onSphere(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         l_hidden=l_hidden,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "vf_fc_vec_ImgToSphere":
    #     in_dim = kwargs['in_dim']
    #     lat_dim = kwargs['lat_dim']
    #     out_dim = kwargs['out_dim']
    #     l_hidden = kwargs["l_hidden"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = vf_FC_vec_ImgToSphere(
    #         in_chan=in_dim,
    #         lat_chan=lat_dim,
    #         out_chan=out_dim,
    #         l_hidden=l_hidden,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "vf_fc_image":
    #     in_dim = kwargs['in_dim']
    #     out_dim = kwargs['out_dim']
    #     l_hidden = kwargs["l_hidden"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = vf_FC_image(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         l_hidden=l_hidden,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "cfw_fc_vec":
    #     in_dim = kwargs['in_dim']
    #     out_dim = kwargs['out_dim']
    #     l_hidden = kwargs["l_hidden"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = cfw_FC_vec(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         l_hidden=l_hidden,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "unet":
    #     image_size = kwargs.get("image_size", 32)
    #     model_channels = kwargs.get("model_channels", 256)
    #     in_channels = kwargs.get("in_channels", 3)
    #     out_channels = kwargs.get("out_channels", 3)
    #     num_res_blocks = kwargs.get("num_res_blocks", 2)
    #     attention_resolutions = kwargs.get("attention_resolutions", [16])
    #     channel_mult = kwargs.get("channel_mult", [1, 2, 2, 2])
    #     num_heads = kwargs.get("num_heads", 4)
    #     num_head_channels = kwargs.get("num_head_channels", 64)
    #     resblock_updown = kwargs.get("resblock_updown", False)
    #     use_scale_shift_norm = kwargs.get("use_scale_shift_norm", False)
    #     use_new_attention_order = kwargs.get("use_new_attention_order", False)
    #     net = UNetModel(
    #         image_size=image_size,
    #         in_channels=in_channels,
    #         model_channels=model_channels,
    #         out_channels=out_channels,
    #         num_res_blocks=num_res_blocks,
    #         attention_resolutions=attention_resolutions,
    #         dropout=0,
    #         channel_mult=channel_mult,
    #         conv_resample=True,
    #         dims=2,
    #         num_classes=None,
    #         use_checkpoint=False,
    #         use_fp16=False,
    #         num_heads=num_heads,
    #         num_head_channels=num_head_channels,
    #         num_heads_upsample=-1,
    #         use_scale_shift_norm=use_scale_shift_norm,
    #         resblock_updown=resblock_updown,
    #         use_new_attention_order=use_new_attention_order,
    #     )
    # elif kwargs["arch"] == "unet2":
    #     input_channels = kwargs.get("input_channels", 4)
    #     output_channels = kwargs.get("output_channels", 4)
    #     model_channels = kwargs.get("model_channels", 64)
    #     channel_mult = kwargs.get("channel_mult", [1,2,4,8,16])
    #     bilinear = kwargs.get("bilinear", False)
    #     net = UNet(
    #         input_channels, 
    #         output_channels, 
    #         model_channels=model_channels, 
    #         channel_mult=channel_mult, 
    #         bilinear=bilinear
    #     )
    # elif kwargs["arch"] == "unet4mapp":
    #     kwargs["arch"] = "unet"
    #     unet = get_net(**kwargs)
    #     net = UNET4MAPP(unet)
    # elif kwargs["arch"] == "unet24mapp":
    #     kwargs["arch"] = "unet2"
    #     unet = get_net(**kwargs)
    #     net = UNET24MAPP(
    #         unet, 
    #         fixed_sigma=kwargs.get('fixed_sigma', False))
    # elif kwargs["arch"] == "conv28":
    #     in_dim = kwargs["in_dim"]
    #     out_dim = kwargs['out_dim']
    #     nh = kwargs["nh"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = ConvNet28(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         nh=nh,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "dconv28":
    #     in_dim = kwargs["in_dim"]
    #     out_dim = kwargs['out_dim']
    #     nh = kwargs["nh"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = DeConvNet28(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         nh=nh,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "conv32":
    #     in_dim = kwargs["in_dim"]
    #     out_dim = kwargs['out_dim']
    #     nh = kwargs["nh"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = ConvNet32(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         nh=nh,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "dconv32":
    #     in_dim = kwargs["in_dim"]
    #     out_dim = kwargs['out_dim']
    #     nh = kwargs["nh"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = DeConvNet32(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         nh=nh,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "conv64":
    #     in_dim = kwargs["in_dim"]
    #     out_dim = kwargs['out_dim']
    #     nh = kwargs["nh"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = ConvNet64(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         nh=nh,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )
    # elif kwargs["arch"] == "dconv64":
    #     in_dim = kwargs["in_dim"]
    #     out_dim = kwargs['out_dim']
    #     nh = kwargs["nh"]
    #     activation = kwargs["activation"]
    #     out_activation = kwargs["out_activation"]
    #     net = DeConvNet64(
    #         in_chan=in_dim,
    #         out_chan=out_dim,
    #         nh=nh,
    #         activation=activation,
    #         out_activation=out_activation,
    #     )

    if kwargs["arch"] == "vf_fc_vec_grasp":
        in_dim = kwargs['in_dim']
        lat_dim = kwargs['lat_dim']
        out_dim = kwargs['out_dim']
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = vf_FC_vec_grasp(
            in_chan=in_dim,
            lat_chan=lat_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "vf_fc_vec_motion":
        in_dim = kwargs['in_dim']
        lat_dim = kwargs['lat_dim']
        out_dim = kwargs['out_dim']
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = vf_FC_vec_motion(
            in_chan=in_dim,
            lat_chan=lat_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == 'dgcnn':
        net = DGCNN(kwargs)
    # elif kwargs["arch"] == 'vn_dgcnn':
    #     net = VN_DGCNN(kwargs)        
    return net


# def get_ae(**model_cfg):
#     if model_cfg["arch"] == "vae":
#         encoder = get_net(**model_cfg["encoder"])
#         decoder = get_net(**model_cfg["decoder"])
#         ae = VAE(encoder, IsotropicGaussian(decoder))
#     elif model_cfg["arch"] == "irvae":
#         reg = model_cfg.get('reg', 1)
#         encoder = get_net(**model_cfg["encoder"])
#         decoder = get_net(**model_cfg["decoder"])
#         ae = IRVAE(encoder, IsotropicGaussian(decoder), reg=reg)
#     return ae


# def get_riemannian_conditional_flow_matching_model(**model_cfg): # added for sphere toy

#     prob_path = model_cfg['prob_path']
#     cnn_pretrained = model_cfg.get('cnn_pretrained', None)

#     device = model_cfg.get('device', None)
#     img_size = model_cfg.get('img_size', [32, 32, 3])

#     if cnn_pretrained is not None:
#         cnn, _ = load_pretrained(**cnn_pretrained)

#     else:
#         model_arch = model_cfg["cnn"].get("arch", None)
#         if model_arch == "cnn32":
#             encoder = ConvNet32(in_chan=3, out_chan=64, nh=32, 
#                                 activation="linear", out_activation="relu")
#             decoder = DeConvNet32(in_chan=64, out_chan=3, nh=32, 
#                                 activation="linear", out_activation="relu")
#             cnn = AE(encoder, decoder)
#         else:
#             cnn = None
#     if cnn == None:
#         model_cfg["velocity_field"]["lat_dim"] = 5
#     velocity_field = get_net(**model_cfg['velocity_field'])

#     model = RCFM_sphere(
#         velocity_field=velocity_field,
#         prob_path=prob_path,
#         cnn_pretrained=cnn,
#         img_size=img_size,
#         device=device,
#     )
#     return model


# def get_riemannian_flow_matching_model(**model_cfg): # added for sphere toy
#     velocity_field = get_net(**model_cfg['velocity_field'])
#     prob_path = model_cfg['prob_path']

#     sigma_1 = model_cfg.get('sigma_1', 0.01)

#     beta_max = model_cfg.get('beta_max', 3)
#     beta_min = model_cfg.get('beta_min', 0.01)
#     beta_type = model_cfg.get('beta_type', 'linear')

#     gpp_pretrained = model_cfg.get('gpp_pretrained', None)
#     mapp_pretrained = model_cfg.get('mapp_pretrained', None)
#     ae_pretrained = model_cfg.get('ae_pretrained', None)

#     val_dl = model_cfg.get('val_dl', None)
#     device = model_cfg.get('device', None)
#     fid = model_cfg.get('fid', False)
#     clipgrad = model_cfg.get('clipgrad', None)

#     z_dim = model_cfg.get('z_dim', 128)
#     img_size = model_cfg.get('img_size', [3, 32, 32])
#     num_interpolants = model_cfg.get('num_interpolants', [1, 8, 1])

#     if gpp_pretrained is not None:
#         gpp, _ = load_pretrained(**gpp_pretrained)
#     else:
#         gpp = None
#     if mapp_pretrained is not None:
#         mapp, _ = load_pretrained(**mapp_pretrained)
#     else:
#         mapp = None
#     if ae_pretrained is not None:
#         ae, _ = load_pretrained(**ae_pretrained)
#     else:
#         ae = None
#     model = RFM_sphere(
#         velocity_field=velocity_field,
#         prob_path=prob_path,
#         sigma_1=sigma_1,
#         beta_max=beta_max,
#         beta_min=beta_min,
#         beta_type=beta_type,
#         gpp_pretrained=gpp,
#         mapp_pretrained=mapp,
#         ae_pretrained=ae,
#         z_dim=z_dim,
#         img_size=img_size,
#         num_interpolants=num_interpolants,        
#         val_dl=val_dl,
#         device=device,
#         fid=fid,
#         clipgrad=clipgrad
#     )
#     return model


# def get_flow_matching_model(**model_cfg):
#     velocity_field = get_net(**model_cfg['velocity_field'])
#     prob_path = model_cfg['prob_path']

#     sigma_1 = model_cfg.get('sigma_1', 0.01)

#     beta_max = model_cfg.get('beta_max', 3)
#     beta_min = model_cfg.get('beta_min', 0.01)
#     beta_type = model_cfg.get('beta_type', 'linear')

#     gpp_pretrained = model_cfg.get('gpp_pretrained', None)
#     mapp_pretrained = model_cfg.get('mapp_pretrained', None)
#     ae_pretrained = model_cfg.get('ae_pretrained', None)

#     val_dl = model_cfg.get('val_dl', None)
#     device = model_cfg.get('device', None)
#     fid = model_cfg.get('fid', False)
#     clipgrad = model_cfg.get('clipgrad', None)

#     z_dim = model_cfg.get('z_dim', 128)
#     img_size = model_cfg.get('img_size', [3, 32, 32])
#     num_interpolants = model_cfg.get('num_interpolants', [1, 8, 1])

#     if gpp_pretrained is not None:
#         gpp, _ = load_pretrained(**gpp_pretrained)
#     else:
#         gpp = None
#     if mapp_pretrained is not None:
#         mapp, _ = load_pretrained(**mapp_pretrained)
#     else:
#         mapp = None
#     if ae_pretrained is not None:
#         ae, _ = load_pretrained(**ae_pretrained)
#     else:
#         ae = None
#     model = FM(
#         velocity_field=velocity_field,
#         prob_path=prob_path,
#         sigma_1=sigma_1,
#         beta_max=beta_max,
#         beta_min=beta_min,
#         beta_type=beta_type,
#         gpp_pretrained=gpp,
#         mapp_pretrained=mapp,
#         ae_pretrained=ae,
#         z_dim=z_dim,
#         img_size=img_size,
#         num_interpolants=num_interpolants,        
#         val_dl=val_dl,
#         device=device,
#         fid=fid,
#         clipgrad=clipgrad
#     )
#     return model


# def get_gpp_model(**model_cfg):
#     module = get_net(**model_cfg['module'])
#     reg = model_cfg.get('reg', 1)
#     alpha_max = model_cfg.get('alpha_max', 1)
#     model = GPP(module, reg=reg, alpha_max=alpha_max)
#     return model


# def get_mapp_model(**model_cfg):
#     module = get_net(**model_cfg['module'])
#     sigma_min = model_cfg.get('sigma_min', 0.01)
#     fixed_sigma = model_cfg.get('fixed_sigma', False)
#     train_mode = model_cfg.get('train_mode', 'default')
#     train_dis_num_t = model_cfg.get('train_dis_num_t', 1000)
#     max_norm = model_cfg.get('max_norm', 0.001)
#     surrogate = model_cfg.get('surrogate', None)
#     num_samples = model_cfg.get('num_samples', 1)
#     num_samples_per_x1 = model_cfg.get('num_samples_per_x1', 2)
#     input_size = model_cfg.get('input_size', None)
#     sampling_per_x1_include = model_cfg.get('sampling_per_x1_include', False)
#     model = MAPP(
#         module, 
#         sigma_min=sigma_min, 
#         fixed_sigma=fixed_sigma, 
#         train_mode=train_mode, 
#         train_dis_num_t=train_dis_num_t,
#         max_norm=max_norm,
#         num_samples=num_samples,
#         surrogate=surrogate,
#         input_size=input_size,
#         num_samples_per_x1=num_samples_per_x1,
#         sampling_per_x1_include=sampling_per_x1_include
#     )
#     return model


# def get_mepp_model(**model_cfg):
#     module = get_net(**model_cfg['module'])
#     sigma_min = model_cfg.get('sigma_min', 0.01)
#     fixed_sigma = model_cfg.get('fixed_sigma', False)
#     train_mode = model_cfg.get('train_mode', 'default')
#     train_dis_num_t = model_cfg.get('train_dis_num_t', 1000)
#     max_norm = model_cfg.get('max_norm', 0.001)
#     num_samples = model_cfg.get('num_samples', 1)
#     num_samples_per_x1 = model_cfg.get('num_samples_per_x1', 2)
#     sampling_per_x1_include = model_cfg.get('sampling_per_x1_include', False)
#     model = MEPP(
#         module, 
#         sigma_min=sigma_min, 
#         fixed_sigma=fixed_sigma, 
#         train_mode=train_mode, 
#         train_dis_num_t=train_dis_num_t,
#         max_norm=max_norm,
#         num_samples=num_samples,
#         num_samples_per_x1=num_samples_per_x1,
#         sampling_per_x1_include=sampling_per_x1_include
#     )
#     return model


# def get_cnn32_model(**model_cfg):
#     if model_cfg["arch"] == "cnn32":
#         encoder = get_net(**model_cfg["encoder"])
#         decoder = get_net(**model_cfg["decoder"])
#         cnn32 = AE(encoder, decoder)
#     else:
#         cnn32 = None
#     return cnn32


# def get_grasp_dif_model(**model_cfg):
#     dim_out_vision_encoder = model_cfg['vision_encoder'].get('dim_out', 132)

#     vision_encoder = VNNResnetPointnet(dim_out_vision_encoder)

#     dim_embed = model_cfg['feature_encoder'].get('dim_embed', 132)
#     dims_hidden_feature_encoder = model_cfg['feature_encoder'].get('dims_hidden_encoder', [512] * 8)
#     num_hidden_layers = len(dims_hidden_feature_encoder)
#     dim_out_feature_encoder = model_cfg['feature_encoder'].get('dim_out_feature_encoder', 7)
#     dropout = model_cfg['feature_encoder'].get('dropout', list(range(num_hidden_layers)))
#     dropout_prob = model_cfg['feature_encoder'].get('dropout_prob', 0.2)
#     norm_layers = model_cfg['feature_encoder'].get('norm_layers', list(range(num_hidden_layers)))
#     latent_in = model_cfg['feature_encoder'].get('latent_in', [4])
#     xyz_in_all = model_cfg['feature_encoder'].get('xyz_in_all', False)
#     use_tanh = model_cfg['feature_encoder'].get('use_tanh', False)
#     latent_dropout = model_cfg['feature_encoder'].get('latent_dropout', False)
#     weight_norm = model_cfg['feature_encoder'].get('weight_norm', True)

#     feature_encoder = TimeLatentFeatureEncoder(
#         dim_embed=dim_embed,
#         dim_out_vision_encoder=dim_out_vision_encoder,
#         dims_hidden=dims_hidden_feature_encoder,
#         dim_out_feature_encoder=dim_out_feature_encoder,
#         dropout=dropout,
#         dropout_prob=dropout_prob,
#         norm_layers=norm_layers,
#         latent_in=latent_in,
#         xyz_in_all=xyz_in_all,
#         use_tanh=use_tanh,
#         latent_dropout=latent_dropout,
#         weight_norm=weight_norm
#     )

#     num_points = model_cfg['points'].get('num_points', 30)
#     location = model_cfg['points'].get('location', [0, 0, 0])
#     scale = model_cfg['points'].get('scale', [1, 1, 1])

#     points = 2 * np.random.rand(num_points, 3) - 1
#     points = points * scale + location
#     points = torch.Tensor(points)

#     dim_in_decoder = num_points * dim_out_feature_encoder
#     dim_hidden_decoder = model_cfg['decoder'].get('dim_hidden_decoder', 512)

#     decoder = EnergyNet(dim_in=dim_in_decoder, dim_hidden=dim_hidden_decoder)

#     if 'steps' in model_cfg:
#         T = model_cfg['steps'].get('T', 70)
#         T_fit = model_cfg['steps'].get('T_fit', 50)
#         k_steps = model_cfg['steps'].get('k_steps', 2)
#         deterministic = model_cfg['steps'].get('deterministic', False)
#     else:
#         T, T_fit, k_steps, deterministic = 70, 50, 2, False

#     model = GraspDiffusionFields(vision_encoder, points, feature_encoder, decoder, T, T_fit, k_steps, deterministic)

#     return model

# def get_grasp_dif_only_model(**model_cfg):
#     dim_out_vision_encoder = model_cfg['vision_encoder'].get('dim_out', 132)

#     vision_encoder = VNNResnetPointnet(dim_out_vision_encoder)

#     dim_embed = model_cfg['feature_encoder'].get('dim_embed', 132)
#     dims_hidden_feature_encoder = model_cfg['feature_encoder'].get('dims_hidden_encoder', [512] * 8)
#     num_hidden_layers = len(dims_hidden_feature_encoder)
#     dim_out_feature_encoder = model_cfg['feature_encoder'].get('dim_out_feature_encoder', 7)
#     dropout = model_cfg['feature_encoder'].get('dropout', list(range(num_hidden_layers)))
#     dropout_prob = model_cfg['feature_encoder'].get('dropout_prob', 0.2)
#     norm_layers = model_cfg['feature_encoder'].get('norm_layers', list(range(num_hidden_layers)))
#     latent_in = model_cfg['feature_encoder'].get('latent_in', [4])
#     xyz_in_all = model_cfg['feature_encoder'].get('xyz_in_all', False)
#     use_tanh = model_cfg['feature_encoder'].get('use_tanh', False)
#     latent_dropout = model_cfg['feature_encoder'].get('latent_dropout', False)
#     weight_norm = model_cfg['feature_encoder'].get('weight_norm', True)

#     feature_encoder = TimeLatentFeatureEncoder(
#         dim_embed=dim_embed,
#         dim_out_vision_encoder=dim_out_vision_encoder,
#         dims_hidden=dims_hidden_feature_encoder,
#         dim_out_feature_encoder=dim_out_feature_encoder,
#         dropout=dropout,
#         dropout_prob=dropout_prob,
#         norm_layers=norm_layers,
#         latent_in=latent_in,
#         xyz_in_all=xyz_in_all,
#         use_tanh=use_tanh,
#         latent_dropout=latent_dropout,
#         weight_norm=weight_norm
#     )

#     num_points = model_cfg['points'].get('num_points', 30)
#     location = model_cfg['points'].get('location', [0, 0, 0])
#     scale = model_cfg['points'].get('scale', [1, 1, 1])

#     points = 2 * np.random.rand(num_points, 3) - 1
#     points = points * scale + location
#     points = torch.Tensor(points)

#     dim_in_decoder = num_points * dim_out_feature_encoder
#     dim_hidden_decoder = model_cfg['decoder'].get('dim_hidden_decoder', 512)

#     decoder = EnergyNet(dim_in=dim_in_decoder, dim_hidden=dim_hidden_decoder)

#     model = GraspDiffusionFieldsONLY(vision_encoder, points, feature_encoder, decoder)

#     return model

def get_grasp_rcfm_model(**model_cfg): # added for grasping
    velocity_field = get_net(**model_cfg['velocity_field'])
    latent_feature = get_net(**model_cfg['latent_feature'])
    prob_path = model_cfg['prob_path']
    init_dist = model_cfg['init_dist']
    ode_solver = model_cfg['ode_solver']
    model = GraspRCFM(
        velocity_field=velocity_field,
        latent_feature=latent_feature,
        prob_path=prob_path,
        init_dist = init_dist,
        ode_solver = ode_solver
    )
    return model

def get_motion_rcfm_model(**model_cfg): # added for motion planning
    velocity_field = get_net(**model_cfg['velocity_field'])
    latent_feature = get_net(**model_cfg['latent_feature'])
    prob_path = model_cfg['prob_path']
    init_dist = model_cfg['init_dist']
    ode_solver = model_cfg['ode_solver']
    model = MotionRCFM(
        velocity_field=velocity_field,
        latent_feature=latent_feature,
        prob_path=prob_path,
        init_dist = init_dist,
        ode_solver = ode_solver
    )
    return model


# def get_grasp_vae_gan_model(**model_cfg):
#     dim_in = model_cfg.get('dim_in', 19)
#     dim_lat = model_cfg.get('dim_lat', 2)
#     if model_cfg['arch'] == 'grasp_vae':
#         model = GraspVAE(dim_in, dim_lat)
#     elif model_cfg['arch'] == 'grasp_gan':
#         model = GraspGAN(dim_in, dim_lat)

#     return model


def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)

    gpp_pretrained_module_root = kwargs.get('gpp_pretrained_module_root', None)
    mapp_pretrained_module_root = kwargs.get('mapp_pretrained_module_root', None)
    cnn32_pretrained_module_root = kwargs.get('cnn32_pretrained_module_root', None)
    if gpp_pretrained_module_root is not None:
        cfg['model']['gpp_pretrained']['root'] = gpp_pretrained_module_root
    if mapp_pretrained_module_root is not None:
        cfg['model']['mapp_pretrained']['root'] = mapp_pretrained_module_root
    if cnn32_pretrained_module_root is not None:
        cfg['model']['cnn_pretrained']['root'] = cnn32_pretrained_module_root ## 사용 안하는듯

    model = get_model(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)

    return model, cfg
