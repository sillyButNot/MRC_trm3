import os
import torch
import timeit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.functions.squad_metric import compute_predictions_logits, squad_evaluate

from transformers import AdamW, get_linear_schedule_with_warmup

from src.functions.utils import load_examples, set_seed, to_list, load_input_data
from transformers.data.processors.squad import SquadResult
from src.functions.evaluate_v1_0 import eval_during_train


def train(args, model, tokenizer, logger):
    # 학습에 사용하기 위한 dataset Load
    train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)

    # tokenizing 된 데이터를 batch size만큼 가져오기위한 random sampler 및 DataLoader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Layer에 따른 가중치 decay 적용
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # optimizer 및 scheduler 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1

    tr_loss, logging_loss = 0.0, 0.0

    # loss buffer 초기화
    model.zero_grad()

    set_seed(args)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # train 모드로 설정
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # 모델에 입력할 입력 tensor 저장
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                # "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            # Loss 계산 및 저장
            outputs = model(**inputs)
            loss = outputs[0]

            # 높은 batch size 효과를 주기위한 gradient_accumulation_step
            # batch size: 16
            # gradient_accumulation_step: 2 라고 가정
            # 실제 batch size 32의 효과와 동일하진 않지만 비슷한 효과를 보임
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step + 1), loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # model save
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 모델 저장 디렉토리 생성
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 학습된 가중치 및 vocab 저장
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Validation Test!!
                    logger.info("***** Eval results *****")
                    results = evaluate(args, model, tokenizer, logger, global_step=global_step)

    return global_step, tr_loss / global_step


# 정답이 사전부착된 데이터로부터 평가하기 위한 함수
def evaluate(args, model, tokenizer, logger, global_step=""):
    # 데이터셋 Load
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)

    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 모델 출력을 저장하기위한 리스트 선언
    all_results = []

    # 평가 시간 측정을 위한 time 변수
    start_time = timeit.default_timer()

    for batch in eval_dataloader:
        # 모델을 평가 모드로 변경
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            # 입력 데이터 별 고유 id 저장 (feature와 dataset에 종속)
            example_indices = batch[3]

            # outputs = (start_logits, end_logits)
            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            # feature 고유 id로 접근하여 원본 q_id 저장
            # 각 feature는 유일한 q_id를 갖고 있지 않음
            # ==> context가 긴 경우, context를 분할하여 여러 개의 데이터로 변환하기 때문!
            eval_feature = features[example_index.item()]

            # 입력 질문에 대한 N개의 결과 저장하기위해 q_id 저장
            unique_id = int(eval_feature.unique_id)

            # outputs = [start_logits, end_logits]
            output = [to_list(output[i]) for output in outputs]

            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            start_logits, end_logits = output

            # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
            result = SquadResult(unique_id, start_logits, end_logits)

            # feature에 종속되는 최종 출력 값을 리스트에 저장
            all_results.append(result)
    # 평가 시간 측정을 위한 time 변수
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # 최종 예측 값을 저장하기 위한 파일 생성
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(global_step))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(global_step))

    # Yes/No Question을 다룰 경우, 각 정답이 유효할 확률 저장을 위한 파일 생성
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(global_step))
    else:
        output_null_log_odds_file = None

    # q_id에 대한 N개의 출력 값의 확률로 부터 가장 확률이 높은 최종 예측 값 저장
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # 최종 예측값과 원문이 저장된 example로 부터 성능 평가 (SQuAD 평가 스크립트)
    results = squad_evaluate(examples, predictions)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    output_dir = os.path.join(args.output_dir, "eval")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # KorQuAD 평가 스크립트 기반 성능 저장을 위한 파일 생성
    output_eval_file = os.path.join(
        output_dir,
        "eval_result_{}_{}.txt".format(list(filter(None, args.model_name_or_path.split("/"))).pop(), global_step),
    )

    logger.info("***** Official Eval results *****")
    with open(output_eval_file, "w", encoding="utf-8") as f:
        # KorQuAD 평가 스크립트 기반의 성능 측정
        official_eval_results = eval_during_train(args, global_step)
        for key in sorted(official_eval_results.keys()):
            logger.info("  %s = %s", key, str(official_eval_results[key]))
            f.write(" {} = {}\n".format(key, str(official_eval_results[key])))
    return results


def predict(args, model, tokenizer):
    model.eval()

    # question = input("Input Question : ")
    # context = input("Input Context : ")
    question = "임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?"
    context = (
        "1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의(폭력행위등처벌에관한법률위반)으로 지명수배되었다. "
        "1989년 3월 12일 서울지방검찰청 공안부는 임종석의 사전구속영장을 발부받았다. "
        "같은 해 6월 30일 평양축전에 임수경을 대표로 파견하여 국가보안법위반 혐의가 추가되었다. "
        "경찰은 12월 18일~20일 사이 서울 경희대학교에서 임종석이 성명 발표를 추진하고 있다는 첩보를 입수했고, 12월 18일 오전 7시 40분 경 가스총과 전자봉으로 무장한 특공조 및 대공과 직원 12명 등 22명의 사복 경찰을 승용차 8대에 나누어 경희대학교에 투입했다. "
        "1989년 12월 18일 오전 8시 15분 경 서울청량리경찰서는 호위 학생 5명과 함께 경희대학교 학생회관 건물 계단을 내려오는 임종석을 발견, 검거해 구속을 집행했다. "
        "임종석은 청량리경찰서에서 약 1시간 동안 조사를 받은 뒤 오전 9시 50분 경 서울 장안동의 서울지방경찰청 공안분실로 인계되었다."
    )
    dataset, examples, features = load_input_data(args, tokenizer, question, context)

    all_results = []
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)
    # 평가 시간 측정을 위한 time 변수
    start_time = timeit.default_timer()

    for batch in eval_dataloader:
        # 모델을 평가 모드로 변경
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            # 입력 데이터 별 고유 id 저장 (feature와 dataset에 종속)
            example_indices = batch[3]

            # outputs = (start_logits, end_logits)
            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            # feature 고유 id로 접근하여 원본 q_id 저장
            # 각 feature는 유일한 q_id를 갖고 있지 않음
            # ==> context가 긴 경우, context를 분할하여 여러 개의 데이터로 변환하기 때문!
            eval_feature = features[example_index.item()]

            # 입력 질문에 대한 N개의 결과 저장하기위해 q_id 저장
            unique_id = int(eval_feature.unique_id)

            # outputs = [start_logits, end_logits]
            output = [to_list(output[i]) for output in outputs]

            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            start_logits, end_logits = output

            # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
            result = SquadResult(unique_id, start_logits, end_logits)

            # feature에 종속되는 최종 출력 값을 리스트에 저장
            all_results.append(result)

    # 최종 예측 값을 저장하기 위한 파일 생성
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format("predict"))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format("predict"))

    # q_id에 대한 N개의 출력 값의 확률로 부터 가장 확률이 높은 최종 예측 값 저장
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        None,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    print(predictions["sample"])
