from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch

# from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel
import torch.nn.functional as F


# from transformers.models.electra import ElectraModel, ElectraPreTrainedModel


class rnn_Decoder(nn.Module):
    def __init__(self, config, hidden_size):
        super(rnn_Decoder, self).__init__()

        # electra의 임베딩을 직접 가져와야하는데 어떻게 해야 가져올 수 있는 거지? -> get_input_embeddings
        self.hidden_size = hidden_size  # 256
        # self.seq_length = config.max_seq_length
        self.vocab_size = config.vocab_size
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # input: (batch, 1, hidden_size //2 ) 16,1,128
        # last_hidden : (1, batch, hidden_size)
        # encoder_outputs : (batch, seq_length, hidden) 16 512 256

        ###########################################################
        # input : (batch, seq,hidden, hidden)
        # last_hidden : (
        batch_size = input.size()[0]

        # rnn_output : (batch, 1, hidden_size)
        # rnn_hidden : (1, batch, hidden_size)
        rnn_output, rnn_hidden = self.gru(input, last_hidden)

        l_rnn_output = self.linear(rnn_output)
        # attention
        # encoder_outputs : (batch, seq_length, hidden) -> (batch, hidden, seq_length)
        encoder_outputs = encoder_outputs.permute(0, 2, 1)

        # attn_weight : (batch, 1, seq_length)
        attn_weights = torch.bmm(l_rnn_output, encoder_outputs)

        # attn_weight =  (batch, 1, seq_length)
        # attn_weights = F.softmax(attn_weights, dim=-1)
        # rnn_output : (batch, 1, hidden)
        return attn_weights, rnn_hidden


class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.cls_number = config.cls_number
        # electra model의 추가 설정
        # output_attentions : 모든 electra layer(12층)의 attention alignment score
        # output_hidden_states : 모든 electra layer(12층)의 attention output
        # 적용 방법
        # config.output_attentions = True
        # config.output_hidden_states = True
        self.vocab_size = config.vocab_size
        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)

        # 정답이 있는 문단인지 확인
        self.cls_linear = nn.Linear(config.hidden_size, 2)
        self.start_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.end_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.decoder_cls = nn.Linear(config.hidden_size, config.hidden_size)
        self.embedding_layer = self.electra.get_input_embeddings()
        self.decoder = rnn_Decoder(config=config, hidden_size=self.hidden_size)

    def forward(
        self,
        input_ids=None,
        ##################
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        ##############
        start_positions=None,
        end_positions=None,
        #################
        is_answer=None,
    ):
        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # sequence_output : [batch_size, seq_length, hidden_size] 16 512 256
        sequence_output = outputs[0]
        cls_outputs = sequence_output[:, 0, :]
        cls_logits = self.cls_linear(cls_outputs)

        ##########################################################################
        # start_logits : (batch, seq_length, hidden)
        # end_logits : (batch, seq_length, hidden)
        # start_logits = self.start_linear(sequence_output)
        # end_logits = self.end_linear(sequence_output)

        ##########################################################################
        # outputs = (start_logits, end_logits, cls_linear)
        outputs = (cls_logits,) + outputs[1:]

        #####################################################
        batch_size = sequence_output.size()[0]
        ###디코더
        # 디코더의 처음 init 값을 초기화? 하는 느낌
        # dedcoder_input : [batch,] (16,) start 심볼을 넣어주는 것
        # 시작 symbol은 cls 토큰을 넣어줌

        # decoder_input : (batch, hidden) -> (batch, 1, hidden)
        decoder_input = cls_outputs.unsqueeze(dim=1)
        # decoder_hidden : batch_size, hidden_size] 16, 128
        # 이게 decoder init 부분
        # decoder_hidden : (1, batch, hidden) 1,16,256
        # decoder_hidden = None

        decoder_hidden = self.decoder_cls(cls_outputs)
        decoder_hidden = decoder_hidden.unsqueeze(dim=0)
        # 디코더 넣을 자리
        # decoder_output : (batch, vocab_size)
        # decoder_hidden : (1, batch, hidden_size)
        # decoder_input : (batch,)

        # attn_weight = (batch, 1, seq_length)
        # decoder_hidden : (1, batch, hidden_size)
        attn_weights, decoder_hidden = self.decoder(
            decoder_input, decoder_hidden, sequence_output
        )
        decoder_start_index = attn_weights
        decoder_start_index = decoder_start_index.squeeze(dim=1)

        attn_weights_softmax = F.softmax(attn_weights, dim=-1)

        # decoder_input : (batch, 1, seq_length) * (batch, seq_length, hidden) = (batch, 1, hidden)
        decoder_input = torch.bmm(attn_weights_softmax, sequence_output)

        # attn_weight = (batch, 1, seq_length)
        # decoder_hidden : (1, batch, hidden_size)
        attn_weights, decoder_hidden = self.decoder(
            decoder_input, decoder_hidden, sequence_output
        )
        decoder_end_index = attn_weights.squeeze(dim=1)

        outputs = (
            decoder_start_index,
            decoder_end_index,
        ) + outputs

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            # ignored_index : max_length
            ignored_index = decoder_start_index.size(1)

            # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # logg_fct 선언
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # start/end에 대해 loss 계산
            start_loss = loss_fct(decoder_start_index, start_positions)
            end_loss = loss_fct(decoder_end_index, end_positions)
            cls_loss = loss_fct(cls_logits, is_answer)

            # 최종 loss 계산
            total_loss = (start_loss + end_loss + cls_loss) / 3

            # outputs : (total_loss, start_logits, end_logits, cls_logits)
            outputs = (total_loss,) + outputs

        return outputs
