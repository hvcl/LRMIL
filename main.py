import argparse
from types import SimpleNamespace

from lrmil.trainer import (
    evaluate,
    train_stage1,
    train_stage2_student,
    train_stage2_teacher,
)


STAGE_RUNNERS = {
    "stage1": train_stage1,
    "stage2_teacher": train_stage2_teacher,
    "stage2_student": train_stage2_student,
    "eval": evaluate,
}


def _int_list(value):
    return [int(x) for x in value.split(",") if x]


def build_config(args):
    return SimpleNamespace(
        experiment=SimpleNamespace(
            name=args.experiment_name,
            seed=args.seed,
            output_dir=args.output_dir,
        ),
        runtime=SimpleNamespace(device=args.device),
        data=SimpleNamespace(
            dataset=args.dataset,
            task=args.task,
            csv_path=args.csv_path,
            lr_patch_root=args.lr_patch_root,
            teacher_aggregated_root=args.teacher_aggregated_root,
            teacher_mil_root=args.teacher_mil_root,
            student_feature_root=args.student_feature_root,
        ),
        model=SimpleNamespace(
            encoder=args.encoder,
            conch_repo=args.conch_repo,
            conch_ckpt=args.conch_ckpt,
            embed_dim=args.embed_dim,
            gated_attention=args.gated_attention,
            num_classes=args.num_classes,
        ),
        training=SimpleNamespace(
            folds=args.folds,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        stage1=SimpleNamespace(
            kd_layers=args.kd_layers,
            lambda_layer=args.lambda_layer,
            lambda_attn=args.lambda_attn,
        ),
        stage2_student=SimpleNamespace(
            temperature_bag=args.temperature_bag,
            temperature_attn=args.temperature_attn,
            lambda_ce=args.lambda_ce,
            lambda_bag=args.lambda_bag,
            lambda_soft=args.lambda_soft,
            lambda_hard=args.lambda_hard,
            topk=args.topk,
            teacher_attn_agg=args.teacher_attn_agg,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="LRMIL training and evaluation entrypoint.")
    parser.add_argument("--stage", choices=sorted(STAGE_RUNNERS))

    parser.add_argument("--experiment_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--dataset", help="Dataset name, e.g. brca, lung, rcc, BRACs.")
    parser.add_argument("--task", help="CSV label column. Use survival for event labels.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--lr_patch_root", help="5x patch tensor root for stage1.")
    parser.add_argument("--teacher_aggregated_root")
    parser.add_argument("--teacher_mil_root", help="Root containing teacher_meta/fold*/. Defaults to output_dir/stage2_teacher.")
    parser.add_argument("--student_feature_root", help="Root containing distilled LR WSI feature .pt files.")

    parser.add_argument("--encoder")
    parser.add_argument("--conch_repo")
    parser.add_argument("--conch_ckpt")
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--gated_attention", action="store_true")
    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--folds", type=_int_list)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)

    parser.add_argument("--kd_layers", type=_int_list)
    parser.add_argument("--lambda_layer", type=float)
    parser.add_argument("--lambda_attn", type=float)

    parser.add_argument("--temperature_bag", type=float)
    parser.add_argument("--temperature_attn", type=float)
    parser.add_argument("--lambda_ce", type=float)
    parser.add_argument("--lambda_bag", type=float)
    parser.add_argument("--lambda_soft", type=float)
    parser.add_argument("--lambda_hard", type=float)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--teacher_attn_agg", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_config(args)
    STAGE_RUNNERS[args.stage](config)


if __name__ == "__main__":
    main()
