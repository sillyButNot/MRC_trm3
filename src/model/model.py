from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from transformers import ElectraModel, ElectraPreTrainedModel

# from transformers.models.electra import ElectraModel, ElectraPreTrainedModel


class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # electra model의 추가 설정
        # output_attentions : 모든 electra layer(12층)의 attention alignment score
        # output_hidden_states : 모든 electra layer(12층)의 attention output
        # 적용 방법
        # config.output_attentions = True
        # config.output_hidden_states = True

        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)
        # final output projection layer(fnn)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # 정답이 있는 문단인지 확인
        self.cls_linear = nn.Linear(config.hidden_size, 2)

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

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]
        cls_outputs = sequence_output[:, 0, :]
        cls_logits = self.cls_linear(cls_outputs)
        # batch, seq_len, 2

        # logits : [batch_size, seq_length, 2]
        logits = self.qa_outputs(sequence_output)

        # start_logits : [batch_size, seq_length, 1]
        # end_logits : [batch_size, seq_length, 1]
        start_logits, end_logits = logits.split(1, dim=-1)

        # start_logits : [batch_size, seq_length]
        # end_logits : [batch_size, seq_length]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # outputs = (start_logits, end_logits, cls_linear)
        outputs = (
            start_logits,
            end_logits,
            cls_logits,
        ) + outputs[1:]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms

            # ignored_index : max_length
            ignored_index = start_logits.size(1)

            # 코드의 안정성을 위해 인덱스 범위 지정 (0~max_length)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # logg_fct 선언
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            # start/end에 대해 loss 계산
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            cls_loss = loss_fct(cls_logits, is_answer)

            # 최종 loss 계산
            # 이거 loss도 물어봐야겠다.....ㅠㅜㅠㅠㅠㅠㅠㅠㅠㅠㅠ
            total_loss = (start_loss + end_loss + cls_loss) / 3

            # outputs : (total_loss, start_logits, end_logits, cls_logits)
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, sent_token_logits
