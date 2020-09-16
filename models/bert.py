import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from models.highway import Highway

# ============== Encoders ===============
class Conv1DEncoder(nn.Module):
    def __init__(self, output_dim=2):
        super(Conv1DEncoder, self).__init__()
        self.output_dim = output_dim
        self.conv1d_1 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, padding=2)
        self.conv1d_3 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, padding=2)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3)
        self.fc = nn.Linear(256, self.output_dim)

    def forward(self, input):
        # permute embeddings - else the model will be destroyed
        input = input.permute(0, 2, 1)
        input = torch.tanh(self.conv1d_1(input))
        input = torch.tanh(self.conv1d_2(input))
        input = torch.tanh(self.conv1d_3(input))
        # back to normal
        input = input.permute(0, 2, 1)
        # TODO: need to activate after max pool?
        input = torch.tanh(self.maxpool_3(input))
        input = self.fc(input)

        return input


class Conv1DEncoder2(nn.Module):
    def __init__(self, output_dim=2):
        super(Conv1DEncoder2, self).__init__()
        self.output_dim = output_dim
        self.conv1d_1 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=5, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=512, out_channels=384, kernel_size=5, padding=2)
        self.conv1d_3 = nn.Conv1d(in_channels=384, out_channels=384, kernel_size=5, padding=2)
        self.conv1d_4 = nn.Conv1d(in_channels=384, out_channels=512, kernel_size=5, padding=2)
        self.conv1d_5 = nn.Conv1d(in_channels=512, out_channels=768, kernel_size=5, padding=2)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3)
        self.fc = nn.Linear(256, self.output_dim)

    def forward(self, input):
        # permute embeddings - else the model will be destroyed
        input = input.permute(0, 2, 1)
        input = torch.tanh(self.conv1d_1(input))
        input = torch.tanh(self.conv1d_2(input))
        input = torch.tanh(self.conv1d_3(input))
        input = torch.tanh(self.conv1d_4(input))
        input = torch.tanh(self.conv1d_5(input))
        # back to normal
        input = input.permute(0, 2, 1)
        # TODO: need to activate after max pool?
        input = torch.tanh(self.maxpool_3(input))
        input = self.fc(input)

        return input


class BiLSTMEncoderDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(BiLSTMEncoderDecoder, self).__init__()
        self.drop_prob = drop_prob
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, input, hidden=None):
        # Apply RNN
        input, hn = self.bilstm(input, hidden)  # (batch_size, seq_len, 2 * hidden_size)
        # Apply dropout (RNN applies dropout after all but the last layer)
        input = F.dropout(input, self.drop_prob, self.training)
        return input, hn

class Conv1DBiLSTM(nn.Module):
    def __init__(self, output_dim=2, drop_prob=0.2):
        super(Conv1DBiLSTM, self).__init__()
        self.output_dim = output_dim
        self.drop_prob = drop_prob

        self.bilstm = BiLSTMEncoderDecoder(input_size=256, hidden_size=768, num_layers=2, drop_prob=self.drop_prob)
        self.conv1d_1 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, padding=2)
        self.conv1d_3 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, padding=2)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3)
        self.fc = nn.Linear(1536, self.output_dim)

    def forward(self, input):
        # Move embeddings through Conv1D
        input = input.permute(0, 2, 1)
        input = torch.relu(self.conv1d_1(input))
        input = torch.relu(self.conv1d_2(input))
        input = torch.relu(self.conv1d_3(input))
        # back to normal
        input = input.permute(0, 2, 1)
        # Pool
        input = torch.relu(self.maxpool_3(input))
        # bilstm
        input, _ = self.bilstm(input)
        # classify
        input = self.fc(input)

        return input

class BiLSTMConvolution(nn.Module):
    def __init__(self, output_dim=2, drop_prob=0.2):
        super(BiLSTMConvolution, self).__init__()
        self.output_dim = output_dim
        self.drop_prob = drop_prob

        self.bilstm = BiLSTMEncoderDecoder(input_size=768, hidden_size=768, num_layers=2, drop_prob=self.drop_prob)
        self.conv1d_1 = nn.Conv1d(in_channels=2304, out_channels=1536, kernel_size=5, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=1536, out_channels=768, kernel_size=5, padding=2)
        self.conv1d_3 = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=5, padding=2)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=3)
        self.fc = nn.Linear(256, self.output_dim)

    def forward(self, input):
        # Move embeddings through BiLSTM
        output, _ = self.bilstm(input)
        # Separate the two-directions contexts
        left_cx = output[:, :, :768]
        right_cx = output[:, :, 768:]
        # Concatenate left and right contexts with the original embeddings
        concat = torch.cat((left_cx, input, right_cx), dim=2)
        # Convolve
        concat = concat.permute(0, 2, 1)
        concat = torch.tanh(self.conv1d_1(concat))
        concat = torch.tanh(self.conv1d_2(concat))
        concat = torch.tanh(self.conv1d_3(concat))
        # back to normal
        concat = concat.permute(0, 2, 1)
        concat = torch.tanh(self.maxpool_3(concat))
        # apply dropout
        concat = F.dropout(concat, p=self.drop_prob, training=self.training)
        # classify
        concat = self.fc(concat)

        return concat


class GRUEncoderDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(GRUEncoderDecoder, self).__init__()
        self.drop_prob = drop_prob
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          bidirectional=False,
                          dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, input, hidden=None):
        # Apply RNN
        input, hn = self.gru(input, hidden)  # (batch_size, seq_len, hidden_size)
        # Apply dropout (RNN applies dropout after all but the last layer)
        input = F.dropout(input, self.drop_prob, self.training)
        return input, hn

# ============== Extensions ===============
class BiLSTMHighway(nn.Module):
    def __init__(self,):
        super(BiLSTMHighway, self).__init__()
        self.bilstm_encoder = BiLSTMEncoderDecoder(input_size=768, hidden_size=768, num_layers=2, drop_prob=0.2)
        self.bilstm_decoder = BiLSTMEncoderDecoder(input_size=768 * 2, hidden_size=768, num_layers=2, drop_prob=0.2)
        self.highway = Highway(size=768 * 2, num_layers=2, f=F.relu)
        # lower the hidden size back to 768 due to bidirectionality
        self.fc = nn.Linear(768 * 2, 2)

    # input is always (batch, seq_len, hidden_size)
    # output should always be ( batch size , seq_len , hidden_size)
    def forward(self, input):
        input, hidden = self.bilstm_encoder(input)
        input = self.highway(input)
        input, _ = self.bilstm_decoder(input, hidden)
        input = F.dropout(input, p=0.2, training=self.training)
        input = self.fc(input)
        return input

class GRUHighway(nn.Module):
    def __init__(self,):
        super(GRUHighway, self).__init__()
        self.gru_encoder = GRUEncoderDecoder(input_size=768, hidden_size=768, num_layers=2, drop_prob=0.2)
        self.gru_decoder = GRUEncoderDecoder(input_size=768, hidden_size=768, num_layers=2, drop_prob=0.2)
        self.highway = Highway(size=768, num_layers=2, f=F.relu)
        self.fc = nn.Linear(768, 2)

    # input is always (batch, seq_len, hidden_size)
    # output should always be ( batch size , seq_len , hidden_size)
    def forward(self, input):
        input, hidden = self.gru_encoder(input)
        input = self.highway(input)
        input, _ = self.gru_decoder(input, hidden)
        input = F.dropout(input, p=0.2, training=self.training)
        input = self.fc(input)
        return input

# ============== Models ===============

class BertExtended(BertPreTrainedModel):
    def __init__(self, config):
        super(BertExtended, self).__init__(config)
        self.bert = BertModel(config)
        # set extension to be None in the default case
        self.extension = None
        # two labels for each token - the probability to be the start and end indices of the answer
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def set_extension(self, extension):
        self.extension = extension

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

        # Check that the extension was set
        if not self.extension:
            raise(RuntimeError("Bert extension was not set!"))

        # The extension must return the proper size for the classification task
        # (batch_size, hidden_features, 2)
        logits = self.extension(sequence_hidden)

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
