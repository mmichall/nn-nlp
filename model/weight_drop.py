import torch
from torch.nn import Parameter


def widget_demagnetizer_y2k_edition(*args, **kwargs):
    # We need to replace flatten_parameters with a nothing function
    # It must be a function rather than a lambda as otherwise pickling explodes
    # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
    # (╯°□°）╯︵ ┻━┻
    return


def _weight_drop(module, weights, dropout):
    # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
    # if issubclass(type(self.module), torch.nn.RNNBase):
    #    self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args)

    setattr(module, 'forward', forward)


def do_nothing(*args, **kwargs):
        return


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class WeightDropLSTM(torch.nn.LSTM):
    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_parameters = do_nothing
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)





