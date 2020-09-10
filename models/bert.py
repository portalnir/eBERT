import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput

# ============== Layers ===============

class GRUEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, dropout=0.0, num_layers=1, bidirectional=False, batch_first=True):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=dropout,
                          num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden

class GRUDecoder(nn.Module):
    def __init__(self, output_size=768, hidden_size=768, dropout=0.0, num_layers=1, bidirectional=False, batch_first=True):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout,
                          num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

# ============== Models ===============

class BertGRUConfig(BertConfig):
    model_type = "bert_gru"
    def __init__(self, *args, **kwargs):
        super(self, BertGRUConfig).__init__(*args, **kwargs)

class BertGRU(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        gru_decoder_out_features = 768
        self.bert = BertModel(config)
        # Encoding
        self.gru_encoder = GRUEncoder(input_size=768, hidden_size=768, dropout=0, num_layers=1, bidirectional=False, batch_first=True)
        # Decoding back
        self.gru_decoder = GRUDecoder(hidden_size=768, output_size=gru_decoder_out_features, batch_first=True)
        # two labels for each token - the probability to be the start and end indices of the answer
        self.qa_outputs = nn.Linear(in_features=gru_decoder_out_features, out_features=2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_hidden, sequence_pooler = outputs
        encoded_out, encoded_hidden  = self.gru_encoder(sequence_hidden)
        decoded_out, decoded_hidden = self.gru_decoder(encoded_out, encoded_hidden)


        logits = self.qa_outputs(decoded_out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
