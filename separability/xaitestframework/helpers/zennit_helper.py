

def get_zennit_attribution_evaluation(composite, net, layer, loader, num_classes, norm_fn, device, mode="max_act"):

    model = net.model

    global LAYER_INPUT
    def fw_hook(module, input, output):

        global LAYER_INPUT
        LAYER_INPUT = input[0]
        return None
    handle = layer.register_forward_hook(fw_hook())

    eye = torch.eye(num_classes, device=device)

    with composite.context(model) as modified:
        for i, (data, labels) in enumerate(loader):

            data_norm = norm_fn(data.to(device))
            data_norm.requires_grad_()

            out = modified(data_norm)

            ins = LAYER_INPUT
            ins.retain_grad()
            attributions = torch.autograd.grad(out, ins, grad_outputs=target)[0]
            attributions = attributions.detach()

    handle.remove()
    return attributions
