import argparse
import os
import logging
from attrdict import AttrDict
import torch
from transformers import ElectraTokenizer, ElectraConfig
from src.model.main_functions import train, evaluate, predict
from src.model.model import ElectraForQuestionAnswering
from src.functions.utils import init_logger, set_seed
from src.functions.processor import SquadResult


def create_model(args):
    # 모델 파라미터 Load
    config = ElectraConfig.from_pretrained(
        args.model_name_or_path
        if args.from_init_weight
        else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
    )

    # tokenizer는 pre-trained된 것을 불러오는 과정이 아닌 불러오는 모델의 vocab 등을 Load
    tokenizer = ElectraTokenizer.from_pretrained(
        args.model_name_or_path
        if args.from_init_weight
        else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        do_lower_case=args.do_lower_case,
    )

    config.cls_number = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    config.max_sentence_number = args.max_sentence_number
    model = ElectraForQuestionAnswering.from_pretrained(
        args.model_name_or_path
        if args.from_init_weight
        else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        config=config,
        from_tf=False,
    )

    # vocab 추가
    # 중요 단어의 UNK 방지 및 tokenize를 방지해야하는 경우(HTML 태그 등)에 활용
    # "세종대왕"이 OOV인 경우 ['세종대왕'] --tokenize-->  ['UNK'] (X)
    # html tag인 [td]는 tokenize가 되지 않아야 함. (완전한 tag의 형태를 갖췄을 때, 의미를 갖기 때문)
    #                             ['[td]'] --tokenize-->  ['[', 't', 'd', ']'] (X)

    if args.from_init_weight and args.add_vocab:
        if args.from_init_weight:
            add_token = {"additional_special_tokens": ["[td]", "추가 단어 1", "추가 단어 2"]}
            # 추가된 단어는 tokenize 되지 않음
            # ex
            # '[td]' vocab 추가 전 -> ['[', 't', 'd', ']']
            # '[td]' vocab 추가 후 -> ['[td]']
            tokenizer.add_special_tokens(add_token)
            model.resize_token_embeddings(len(tokenizer))
    model.cuda()
    return model, tokenizer


def main(cli_args):
    # 파라미터 업데이트
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    logger = logging.getLogger(__name__)

    # logger 및 seed 지정
    init_logger()
    set_seed(args)

    # 모델 불러오기
    model, tokenizer = create_model(args)

    # Running mode에 따른 실행
    if args.do_train:
        train(args, model, tokenizer, logger)
    elif args.do_eval:
        evaluate(args, model, tokenizer, logger)
    elif args.do_predict:
        predict(args, model, tokenizer)


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    # Directory
    # cli_parser.add_argument("--model_name_or_path", type=str, default="../MRC_trm3/init_weight") #monologg/koelectra-base-v2-discriminator
    cli_parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="../MRC_trm3/koelectra-small-v2-discriminator",
    )

    cli_parser.add_argument("--output_dir", type=str, default="./test_sentence")

    # Model Hyper Parameter
    cli_parser.add_argument("--max_seq_length", type=int, default=512)
    cli_parser.add_argument("--doc_stride", type=int, default=128)
    cli_parser.add_argument("--max_query_length", type=int, default=64)

    cli_parser.add_argument("--max_answer_length", type=int, default=30)
    cli_parser.add_argument("--n_best_size", type=int, default=20)

    #!!! add sentence max index
    cli_parser.add_argument("--max_sentence_number", type=int, default=160)

    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=5e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default=16)
    cli_parser.add_argument("--eval_batch_size", type=int, default=32)
    cli_parser.add_argument("--num_train_epochs", type=int, default=10)

    cli_parser.add_argument("--save_steps", type=int, default=2000)
    cli_parser.add_argument("--logging_steps", type=int, default=2000)

    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--threads", type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    cli_parser.add_argument("--do_lower_case", type=bool, default=False)

    # For SQuAD v2.0 (Yes/No Question)
    cli_parser.add_argument("--version_2_with_negative", type=bool, default=False)
    cli_parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    # Running Mode
    ######
    cli_parser.add_argument("--data_dir", type=str, default="../MRC_trm3/data")
    cli_parser.add_argument("--train_file", type=str, default="KorQuAD_v1.0_train.json")
    cli_parser.add_argument("--predict_file", type=str, default="KorQuAD_v1.0_dev.json")
    cli_parser.add_argument("--checkpoint", type=str, default="6000")

    # 언어 모델링 학습만 완료된 모델 가중치를 사용할것인지 (True)
    # finetune 을 통해 저장된 모델을 사용할 것인지 (False)
    cli_parser.add_argument("--from_init_weight", type=bool, default=True)
    cli_parser.add_argument("--add_vocab", type=bool, default=False)

    # 저희가 갖고있는 데이터를 사용
    cli_parser.add_argument("--do_train", type=bool, default=True)
    cli_parser.add_argument("--do_eval", type=bool, default=True)

    #
    cli_parser.add_argument("--do_predict", type=bool, default=False)
    ######
    cli_args = cli_parser.parse_args()
    main(cli_args)

    # for check_point_value in range(20000, 42000, 2000):
    #     cli_args.checkpoint = str(check_point_value)
    #     print(cli_args.checkpoint)
    #     main(cli_args)

    # cls vector로 answerable sentence 에서 정답 찾기
