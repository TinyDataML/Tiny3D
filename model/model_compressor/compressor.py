import torch.nn.utils.prune as prune
import torch

def torch_prune(model, prune_list, amount_list):

    for n, module in model.named_modules():
        for j, prune_item in enumerate(prune_list):
            # print(j)
            # print(prune_item)
            if isinstance(module, prune_item):
                prune.random_unstructured(module,name = 'weight', amount = amount_list[j])
                prune.remove(module,'weight')

def dynamic_quant(model):
    model.cpu()
    torch.quantization.quantize_dynamic(model, {torch.nn.Linear},  dtype=torch.qint8)
    model.cuda()

def static_quant(model, input_data):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    model_prepared(*input_data)
    model_prepared.cpu()
    model_int8 = torch.quantization.convert(model_prepared, inplace=True)

    return model_int8
    # torch_out = model_int8(*input_data)