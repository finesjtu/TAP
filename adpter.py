import torch.nn as nn
from transformers.activations import get_activation
from dataclasses import dataclass
from transformers.adapters import RobertaAdapterModel, AdapterConfig


from functools import reduce
import torch
import numpy as np
# import log

# logger = log.get_logger('root')

BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
    'all': 'bias',
    'lm': 'lm_head',
    'ada': 'adapter',
    'layer_norm': 'LayerNorm',
}


@dataclass
class AdapterConfig_(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    # This is for the layernorms applied after feedforward/self-attention layers.
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = False
    nonlinearity: str = "gelu_new"
    reduction_factor: int = 16
    # By default, we add adapters after attention, set False if otherwise.
    add_adapter_after_attention = True
    add_adapter_after_feedforward = True
    # Trains the adapters if this is set to true.
    adapter_tune = True


class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    """Conventional adapter latyer."""
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.activation = Activations(config.nonlinearity.lower())
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)

    def forward(self, x):
        output = self.down_sampler(x)
        output = self.activation(output)
        return self.up_sampler(output)


class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logits of
    putting adapter layers within  the transformer's layers.
    config: adapter configuraiton.
    input_dim: input dimension of the hidden representation feed into adapters."""
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.adapter = self.construct_adapters()
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(input_dim)

    def construct_adapters(self):
        """Construct the Adapter layers."""
        return Adapter(self.config, input_dim=self.input_dim)

    def forward(self, inputs):
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        outputs = self.adapter(z)
        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        return outputs


class RobertaSelfOutput(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        adapter_tune = True if (adapter_config is not None and adapter_config.adapter_tune) else False
        self.add_adapter_after_attention = adapter_tune and adapter_config.add_adapter_after_attention
        if self.add_adapter_after_attention:
            self.self_attention_adapter = AdapterController(adapter_config, input_dim=config.hidden_size)



    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.add_adapter_after_attention:
            hidden_states = self.self_attention_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        adapter_tune = True if (adapter_config is not None and adapter_config.adapter_tune) else False
        self.add_adapter_after_feedforward = adapter_tune and adapter_config.add_adapter_after_feedforward
        if self.add_adapter_after_feedforward:
            self.feed_forward_adapter = AdapterController(adapter_config, input_dim=config.hidden_size)


    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.add_adapter_after_feedforward:
            hidden_states = self.feed_forward_adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Roberta_Adapter(nn.Module):

    def __init__(self, model, model_config, encoder_trainable=True, trainable_components=None, model_restructure=False,
                 ):
        super(Roberta_Adapter, self).__init__()
        self.adapter_config = AdapterConfig_
        self.model_config = model_config
        self.Adapter_list = []
        self.masks = dict()
        # self.model_config = RobertaConfig.from_pretrained(
        #     'roberta-large', num_labels=len(['1','2','3']), finetuning_task='test')

        # self.roberta = RobertaForMaskedLM.from_pretrained('roberta-large', config=self.model_config)
        self.roberta = model

        for i in range(24):
            self.Adapter_list.append([RobertaSelfOutput(self.model_config, self.adapter_config),
                                      RobertaOutput(self.model_config, self.adapter_config)])
        if model_restructure:
            self.model_restructure()

        components = self.convert_to_actual_components(trainable_components)


        if encoder_trainable:
            self._deactivate_relevant_gradients(components)


    @staticmethod
    def convert_to_actual_components(components):
        return [BIAS_TERMS_DICT[component] for component in components]

    def get_structure(self):
        print(self.roberta)


    def get_params(self):
        for name,param in self.roberta.named_parameters():
            print(name, param)

    def model_restructure(self):
        # print(self.attention[0])
        # print(self.roberta.children())
        for i in range(24):
            self.roberta.roberta.encoder.layer[i].attention.output = self.Adapter_list[i][0]
            # print(self.selfoutput)
            self.roberta.roberta.encoder.layer[i].output = self.Adapter_list[i][1]

    def forward(self, **input):
        output = self.roberta(**input)
        return output


    def _deactivate_relevant_gradients(self, trainable_components):
        """Turns off the model parameters requires_grad except the trainable_components.
        Args:
            trainable_components (List[str]): list of trainable components (the rest will be deactivated)
        """
        for param in self.roberta.parameters():
            param.requires_grad = False
        if trainable_components:
            trainable_components = trainable_components + ['pooler.dense.bias']
        trainable_components = trainable_components + ['classifier']
        # trainable_components = trainable_components + ['adapter']
        # trainable_components = trainable_components+ ['lm_head']
        for name, param in self.roberta.named_parameters():
            for component in trainable_components:
                if component in name:
                    param.requires_grad = True
                    break

    def trainable_params_count(self):

        total_params = 0
        trainable_params = 0
        for name, param in self.roberta.named_parameters():

            total_params += reduce(lambda x, y: x * y, param.shape)
            if param.requires_grad:
                trainable_params += reduce(lambda x, y: x * y, param.shape)
                print(name)


        print(f'trainable params amount: {trainable_params}. '
                    f'Total params: {total_params}, Ratio_(trainable_params/total_params): {trainable_params/total_params}')


class Roberta_Adapter_1(nn.Module):
    def __init__(self, encoder_trainable=True, trainable_components=None, activate_adapter=True,
                 deactivate_trainable_para=False):
        super(Roberta_Adapter_1, self).__init__()
        self.model = RobertaAdapterModel.from_pretrained('roberta-large')
        # print(self.model)
        self.config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
        if activate_adapter:
            self.model.add_adapter("bottleneck_adapter", config=self.config)
            #set_active=True
        self.model.add_masked_lm_head('lm_head')
        self.components = trainable_components


        if encoder_trainable and deactivate_trainable_para:
            raise ValueError('Can not activate and deactivate simultaneously!')

        if encoder_trainable:
            trainable_components = self.convert_to_actual_components(trainable_components)
            self._activate_relevant_gradients(trainable_components)




        if deactivate_trainable_para:
            self._deactivate_relevant_gradients(self.components)

        self.trainable_params_count()

    @staticmethod
    def convert_to_actual_components(components):
        return [BIAS_TERMS_DICT[component] for component in components]

    # @staticmethod
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        # torch.save(self.model.state_dict(), path + '/model_parameters.bin')

    def _activate_relevant_gradients(self, trainable_components):
        """Turns off the model parameters requires_grad except the trainable_components.
        Args:
            trainable_components (List[str]): list of trainable components (the rest will be deactivated)
        """
        for param in self.model.parameters():
            param.requires_grad = False
        # if trainable_components:
        #     trainable_components = trainable_components + ['pooler.dense.bias']
        # trainable_components = trainable_components + ['classifier']
        for name, param in self.model.named_parameters():
            for component in trainable_components:
                if component in name:
                    param.requires_grad = True
                    break

    def _deactivate_relevant_gradients(self, trainable_components):
        """Turns off the model parameters requires_grad except the trainable_components.
        Args:
            trainable_components (List[str]): list of trainable components (the rest will be deactivated)
        """
        for param in self.model.parameters():
            param.requires_grad = True
        # if trainable_components:
        #     trainable_components = trainable_components
        for name, param in self.model.named_parameters():
            for component in trainable_components:
                if component in name:
                    param.requires_grad = False
                    break

    def forward(self, **input):
        output = self.model(**input)
        return output

    def trainable_params_count(self):

        total_params = 0
        trainable_params = 0
        for name, param in self.model.named_parameters():

            total_params += reduce(lambda x, y: x * y, param.shape)
            if param.requires_grad:
                trainable_params += reduce(lambda x, y: x * y, param.shape)
                # print(name, param)


        print(f'trainable params amount: {trainable_params}. '
                    f'Total params: {total_params}, Ratio_(trainable_params/total_params): {trainable_params/total_params}')

if __name__ == '__main__':
    # config = AdapterConfig_
    # print(config.nonlinearity)
    # roberta_config = RobertaConfig.from_pretrained('roberta-large', num_labels=len(['1','2','3']), finetuning_task='test')
    # model = RobertaForMaskedLM.from_pretrained('roberta-large', config=roberta_config)
    # model = Roberta_Adapter(model=model, model_config=roberta_config, trainable_components=['ada', 'layer_norm', 'lm'])
    # # print(model)
    # model.get_structure()
    # model.trainable_params_count()

    model = Roberta_Adapter_1(trainable_components=['all', 'ada', 'layer_norm'], activate_adapter=True)
    print(model)
    model.trainable_params_count()