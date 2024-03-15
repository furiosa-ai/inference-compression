import torch


def set_optimization(args):
    if args.torch_optim == 'default':
        return
    elif args.torch_optim == 'none':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, 'opt_einsum'):
            torch.backends.opt_einsum.enabled = False
    else:
        raise ValueError(f"Wrong argument value for '--torch_optim': {args.torch_optim}")

    return
