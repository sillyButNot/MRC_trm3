# MRC_trm3
main
-> baseline

baseline
-> 기존 MRC 코드임
test_baseline

ver.1
-> CLS 를 통해 정답이 있는 문단인지 아닌지를 확인 + CLS 확률이 높은 것 중에서 start_logit과 end_logit 의 합이 높은 것을 정답으로 함
뭐였는지 기억 안남.. 찾아봐야함

ver.2
-> pointer network 를 사용하여 attention을 사용하였음
test_decoder_cls_input
!!!실험 다시 해야함 cls logits 삭제 해야함

ver.3
-> sentence attention 사용
test_sentence 폴더에 결과가 있음

ver.4
중간에 end 쪽 decoder input 하는 부분에서 문장단위만 넣어줌

ver.5
ver.4에서 문장단위만 넣는 것이 이상함, start_index 값을 보고 추론할 수 있도록 하기 위해 rnn_decoder 를 사용하는 것이기 때문에 start_index 관련 내용을 concat 해서 넣어줌

ver.6
-> decoder 초기 input 부터 context 벡터를 따로 만들어서 시작
