from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch

# from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.models.electra import ElectraModel, ElectraPreTrainedModel
import torch.nn.functional as F


# from transformers.models.electra import ElectraModel, ElectraPreTrainedModel


class attn_sequence(nn.Module):
    def __init__(self, hidden_size):
        super(attn_sequence, self).__init__()
        self.hidden_size = hidden_size
        self.sentence_linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, query, key, sentence_mask=None):
        attn_weight = torch.bmm(query, key.transpose(1, 2))

        if sentence_mask != None:
            attn_score = F.softmax(attn_weight + sentence_mask, dim=-1)
        else:
            attn_score = F.softmax(attn_weight, dim=-1)
        # decoder_input : [batch, 1, seq_length] * [batch, seq_length, hidden] = [batch, 1, hidden]
        value = torch.bmm(attn_score, key)

        # decoder_input : [batch, 1, hidden]
        # decoder_index : [batch, hidden]
        return attn_weight, attn_score, value


class rnn_Decoder(nn.Module):
    def __init__(self, config, hidden_size):
        super(rnn_Decoder, self).__init__()

        # electra의 임베딩을 직접 가져와야하는데 어떻게 해야 가져올 수 있는 거지? -> get_input_embeddings
        self.hidden_size = hidden_size  # 256
        # self.seq_length = config.max_seq_length
        self.vocab_size = config.vocab_size
        self.gru = nn.GRU(
            input_size=2 * self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, last_hidden):
        # input : (batch, seq,hidden, hidden) -> (batch, 1, 2 * hidden)
        # last_hidden : (
        batch_size = input.size()[0]
        # rnn_output : (batch, 1, hidden_size)
        # rnn_hidden : (1, batch, hidden_size)
        rnn_output, rnn_hidden = self.gru(input, last_hidden)

        # l_rnn_output : [batch, 1, hidden_size]
        l_rnn_output = self.linear(rnn_output)
        # attention

        # l_rnn_output =  (batch, 1, seq_length)
        # rnn_hidden : (1, batch, hidden_size)
        return l_rnn_output, rnn_hidden


class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)

        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.cls_number = config.cls_number
        self.max_sentence_number = config.max_sentence_number

        # electra model의 추가 설정
        # output_attentions : 모든 electra layer(12층)의 attention alignment score
        # output_hidden_states : 모든 electra layer(12층)의 attention output
        # 적용 방법
        # config.output_attentions = True
        # config.output_hidden_states = True
        self.vocab_size = config.vocab_size

        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)

        self.start_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.end_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.decoder_cls = nn.Linear(config.hidden_size, config.hidden_size)

        self.embedding_layer = self.electra.get_input_embeddings()
        self.decoder = rnn_Decoder(config=config, hidden_size=self.hidden_size)
        self.attn_sequence = attn_sequence(self.hidden_size)

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
        sentence_map=None,
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

        ##########################################################################
        # start_logits : (batch, seq_length, hidden)
        # end_logits : (batch, seq_length, hidden)
        # start_logits = self.start_linear(sequence_output)
        # end_logits = self.end_linear(sequence_output)

        ##########################################################################
        # outputs = (start_logits, end_logits, cls_linear)

        #####################################################
        batch_size = sequence_output.size()[0]
        ###디코더
        # 디코더의 처음 init 값을 초기화? 하는 느낌
        # dedcoder_input : [batch,] (16,) start 심볼을 넣어주는 것
        # 시작 symbol은 cls 토큰을 넣어줌
        # !!!sentence attention 코드 추가
        # sentence_one_hot : [batch, seq_length, sentence_number]
        sentence_one_hot = F.one_hot(sentence_map, num_classes=self.max_sentence_number)
        sentence_one_hot[sentence_map == 0] = 0
        # sentence_one_hot : [batch, seq_length, sentence_number] -> [batch, sentence_number, seq_length]
        sentence_one_hot = sentence_one_hot.float().transpose(1, 2)
        # sentence_representation :[batch, sentence_number, seq_length] * [batch_size, seq_length, hidden_size]
        # = [batch, sentence_number, hidden_size]

        if sequence_output.device != sentence_one_hot.device:
            sentence_one_hot = sentence_one_hot.to(sequence_output.device)

        sentence_representation = torch.bmm(sentence_one_hot, sequence_output)

        # sentence_representation : (batch, sentence_number, hidden)
        # 더해서 0이 되는 부분은 애초에 패딩일 것임.
        # sentence_mask : (batch, sentence_number)
        sentence_mask = torch.sum(sentence_representation, dim=-1)

        sentence_mask_result = sentence_mask.masked_fill(sentence_mask == 0, float("-inf")).masked_fill(
            sentence_mask != 0, 0
        )

        if sequence_output.device != sentence_mask_result.device:
            sentence_mask_result = sentence_mask_result.to(sequence_output.device)
        #################################################################################
        #################################################################################
        #                                          start
        #################################################################################
        #################################################################################

        # decoder_input : (batch, hidden) -> (batch, 1, hidden)
        # decoder inpujt을 CLS 벡터와 sentence_represent 와 attention 한 값을 사용
        # context_weight : (batch, 1, hidden) * (batch, hidden,  sentence_number) -> (batch, 1, sentence_number)

        # cls_outputs : (batch, hidden) -> (batch, 1, hidden)
        # sentence_representation : (batch, sentence_number, hidden)
        # context_weight, context_score : (batch, 1, sentence_number)
        # context_vecotr : (batch, 1, hidden)
        context_weight, context_score, context_vector = self.attn_sequence(
            cls_outputs.unsqueeze(dim=1), sentence_representation, sentence_mask_result.unsqueeze(dim=1)
        )
        # start_sentence : (batch, 3)
        _, start_sentence = context_score.squeeze(dim=1).topk(3, dim=-1)
        start_end_sum_logits = context_weight.squeeze(dim=1)

        #!!!decoder_input = (bathc, 1, 2* hidden)
        decoder_input = torch.cat((cls_outputs.unsqueeze(dim=1), context_vector), dim=-1)
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

        # start_rnn_output : (batch, 1, hidden)
        # decoder_hidden : (1, batch, hidden_size)
        # start_attn_weights : [batch, 1, sentence_number]
        start_rnn_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        # attn_rnn_electra : (batch, 1, hidden)
        # attn_score : (batch, 1, seq_length)

        attn_rnn_electra_weight, attn_rnn_electra_score, attn_rnn_electra = self.attn_sequence(
            start_rnn_output, sequence_output
        )

        # decoder_start_index : (batch, seq_length)
        decoder_start_index = attn_rnn_electra_weight.squeeze(dim=1)
        # attn_rnn_electra : (batch, 1, hidden)
        # sentence_representation = [batch, sentence_number, hidden_size]
        # attn_token_sentence_for_end : (batch, 1, hidden)
        # attn_token_sentence_for_end_score : (batch, 1, sentence_number)

        (
            attn_token_sentence_for_end_weight,
            attn_token_sentence_for_end_score,
            attn_token_sentence_for_end,
        ) = self.attn_sequence(attn_rnn_electra, sentence_representation, sentence_mask_result.unsqueeze(dim=1))

        # end_sentence : (batch, 3)
        _, end_sentence = attn_token_sentence_for_end_score.squeeze(dim=1).topk(3, dim=-1)
        # start_end_sum_logits : (batch, max_length)
        start_end_sum_logits = start_end_sum_logits + attn_token_sentence_for_end_weight.squeeze(dim=1)

        _, start_end_sum_logits = (start_end_sum_logits + sentence_mask_result).topk(3, dim=-1)

        #################################################################################
        #################################################################################
        #                                          end
        #################################################################################
        #################################################################################

        # end_rnn에 input 해줄 것
        # decoder_input : (batch, 1, 2*hidden)
        decoder_input = torch.cat((attn_rnn_electra, attn_token_sentence_for_end), dim=-1)

        # attn_sentence_output : (batch, 1, hidden)
        # decoder_hidden : (1, batch, hidden_size)
        end_rnn_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        # attn_rnn_electra : (batch, 1, hidden)
        # attn_score : (batch, 1, seq_length)
        attn_rnn_electra_weight, attn_rnn_electra_score, attn_rnn_electra = self.attn_sequence(
            end_rnn_output, sequence_output
        )

        # decoder_end_index : (batch, seq_length)
        decoder_end_index = attn_rnn_electra_weight.squeeze(dim=1)

        # outputs : (decoder_start_index, decoder_end_index)
        outputs = (
            decoder_start_index,
            decoder_end_index,
        )

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
            # loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction="none")

            # start/end에 대해 loss 계산
            start_loss = loss_fct(decoder_start_index, start_positions)
            end_loss = loss_fct(decoder_end_index, end_positions)

            # 최종 loss 계산
            total_loss = (start_loss + end_loss) / 2

            # outputs : (total_loss, start_logits, end_logits)
            outputs = (total_loss,) + outputs
        else:
            # outputs : (start_sentence, end_sentence, start_logits, end_logits)
            outputs = (
                start_sentence,
                end_sentence,
                start_end_sum_logits,
            ) + outputs

        return outputs
