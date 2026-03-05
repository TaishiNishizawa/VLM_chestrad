import torch
from torch.utils.data import DataLoader
import os, argparse, time
from mimicvlm.data.mimic_dataset import MimicCXRDataset, collate_skip_none
from mimicvlm.utils.seed import set_seed, seed_worker
from mimicvlm.training.metrics import compute_multilabel_metrics, log_per_label_metrics
from mimicvlm.models.encoders.biomedclip import BiomedCLIP
from mimicvlm.models.heads.mlp_head import MLPHead
from mimicvlm.models.loss import BCEWithLogitsConfig, MultiLabelBCEWithLogits
from mimicvlm.utils.io import ensure_dir, append_row_csv, write_json
from mimicvlm.training.baseline import (
    freeze_encoder,
    run_one_epoch_baseline,
    evaluate_baseline,
    run_one_epoch_head_only,
    evaluate_head_only,
    tune_threshold_on_val
)
from mimicvlm.data.embedding_dataset import EmbeddingShardDataset
from mimicvlm.data.constants import CHEXPERT_LABELS_14


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_jpg_root", type=str, required=True)
    
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_dir", type=str, default="artifacts/biomedclip_mlp")
    ap.add_argument("--pos_weight", type=float, nargs=14, default=[1.0] * 14)

    ap.add_argument("--embeddings_dir", type=str, default=None)
    ap.add_argument("--use_cached_embeddings", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    print("cuda visible:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch cuda device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

    save_dir = ensure_dir(args.save_dir)
    write_json(vars(args), os.path.join(save_dir, "args.json"))
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.use_cached_embeddings:
        assert args.embeddings_dir is not None, "embeddings_dir must be provided when using cached embeddings"

        train_ds = EmbeddingShardDataset(os.path.join(args.embeddings_dir, "train"))
        val_ds = EmbeddingShardDataset(os.path.join(args.embeddings_dir, "validate"))
        test_ds = EmbeddingShardDataset(os.path.join(args.embeddings_dir, "test"))
        train_dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,          # can be low now
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        val_dataloader = DataLoader(
            val_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

        test_dataloader = DataLoader(
            test_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
        encoder = None 
        in_dim = 512
    else:
        encoder = BiomedCLIP().to(device)
        freeze_encoder(encoder)
        # torch.backends.cudnn.benchmark = True
        in_dim = encoder.embed_dim
        train_ds = MimicCXRDataset(
            mimic_cxr_jpg_root = args.mimic_cxr_jpg_root,
            split="train",
            transform=encoder.preprocess,
            label_policy="uncertain_as_negative",
            bad_image_log=os.path.join(save_dir, "bad_images_train.tsv"),
        )
        val_ds = MimicCXRDataset(
            mimic_cxr_jpg_root = args.mimic_cxr_jpg_root,
            split="validate",
            transform=encoder.preprocess,
            label_policy="uncertain_as_negative",
            bad_image_log=os.path.join(save_dir, "bad_images_val.tsv"),
        )
        test_ds = MimicCXRDataset(
            mimic_cxr_jpg_root = args.mimic_cxr_jpg_root,
            split="test",
            transform=encoder.preprocess,
            label_policy="uncertain_as_negative",
            bad_image_log=os.path.join(save_dir, "bad_images_test.tsv"),
        )

        print(f"Size of datasets: ---------\n")
        print(f"Train: {len(train_ds)} | Validation: {len(val_ds)}")
        g = torch.Generator().manual_seed(args.seed)

        train_dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_skip_none,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        val_dataloader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_skip_none,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        test_dataloader = DataLoader(
            test_ds,
            batch_size=8,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_skip_none,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    
    head = MLPHead(
        in_dim=in_dim,
        out_dim=14,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pos_weight = torch.tensor(args.pos_weight, device=device)
    criterion = MultiLabelBCEWithLogits(BCEWithLogitsConfig(pos_weight=pos_weight))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    # --- Train
    best_val = float("-inf")
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    metrics_csv = save_dir / "metrics.csv"

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()

        # ---- training
        head.train()
        train_t0 = time.perf_counter()
        if args.use_cached_embeddings:
            train_loss = run_one_epoch_head_only(
                head=head,
                criterion=criterion,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                amp=True
            )
        else:
            train_loss = run_one_epoch_baseline(
                encoder=encoder,
                head=head,
                criterion=criterion,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                amp=True
            )
        torch.cuda.synchronize()
        train_sec = time.perf_counter() - train_t0

        # ---- validation
        head.eval()
        val_t0 = time.perf_counter()
        if args.use_cached_embeddings:
            val_res = evaluate_head_only(
                head=head,
                criterion=criterion,
                dataloader=val_dataloader,
                device=device,
                compute_metrics_fn=compute_multilabel_metrics,
            )
        else:
            val_res = evaluate_baseline(
                encoder=encoder,
                head=head,
                criterion=criterion,
                dataloader=val_dataloader,
                device=device,
                compute_metrics_fn=compute_multilabel_metrics,
            )
        scheduler.step()
        torch.cuda.synchronize()
        val_sec = time.perf_counter() - val_t0
        epoch_sec = time.perf_counter() - epoch_t0

        val_loss = float(val_res.loss)
        val_metrics = val_res.metrics or {}

        lr = float(optimizer.param_groups[0]["lr"])
        max_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

        val_macro_auroc = float(val_metrics.macro_auroc)
        val_macro_f1 = float(val_metrics.macro_f1)

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_macro_auroc={val_macro_auroc:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} "
            f"time(train/val/epoch)={train_sec:.1f}/{val_sec:.1f}/{epoch_sec:.1f}s "
            f"lr={lr:.2e} "
            f"max_mem={max_mem_mb:.0f}MB"
        )

        # ---- CSV logging (one row per epoch)
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": val_loss,
            "val_macro_auroc": val_macro_auroc,
            "val_macro_f1": val_macro_f1,
            "lr": lr,
            "train_seconds": float(train_sec),
            "val_seconds": float(val_sec),
            "epoch_seconds": float(epoch_sec),
            "train_size": int(len(train_ds)),
            "val_size": int(len(val_ds)),
            "max_memory_allocated_mb": max_mem_mb,
        }
        append_row_csv(metrics_csv, row)

        # ---- save last every epoch (resume-friendly)
        torch.save(
            {
                "epoch": int(epoch),
                # "encoder_model_id": encoder.model_id,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "embed_dim": in_dim,
                "val_macro_auroc": val_macro_auroc,
                "args": vars(args),
            },
            last_path,
        )

        # ---- save best (guard against nan)
        score = val_macro_auroc
        if score == score and score > best_val:  # (score==score) is a cheap isnan check
            best_val = float(score)
            torch.save(
                {
                    "epoch": int(epoch),
                    # "encoder_model_id": encoder.model_id,
                    "head_state_dict": head.state_dict(),
                    "embed_dim": in_dim,
                    "val_macro_auroc": float(best_val),
                    "args": vars(args),
                },
                best_path,
            )

    print(f"Saved best checkpoint to: {best_path} (macro_auroc={best_val:.4f})")
    print(f"Saved metrics CSV to: {metrics_csv}")


    print("------Test Dataset Performance...")
    head = MLPHead(in_dim=in_dim, out_dim=14, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
    head.load_state_dict(torch.load(best_path, weights_only=True)["head_state_dict"])
    head.to(device)
    head.eval()

    print("Tuning F1 thresholds on validation set...")
    # Tune threshold on val set
    tuned_threshold = tune_threshold_on_val(head, val_dataloader, device)
    for name, t in zip(CHEXPERT_LABELS_14, tuned_threshold):
        print(f"  {name:35s} threshold={t:.2f}")
    
    if args.use_cached_embeddings:
        test_result_default = evaluate_head_only(
            head=head,
            criterion=criterion,
            dataloader=test_dataloader,
            device=device,
            compute_metrics_fn=compute_multilabel_metrics,
            threshold=0.5
        )
        test_result_tuned = evaluate_head_only(
            head=head,
            criterion=criterion,
            dataloader=test_dataloader,
            device=device,
            compute_metrics_fn=compute_multilabel_metrics,
            threshold=tuned_threshold
        )
        
    else:
        test_result_default = evaluate_baseline(
            head=head,
            criterion=criterion,
            dataloader=test_dataloader,
            device=device,
            compute_metrics_fn=compute_multilabel_metrics,
            threshold=0.5
        )
        test_result_tuned = evaluate_baseline(
            head=head,
            criterion=criterion,
            dataloader=test_dataloader,
            device=device,
            compute_metrics_fn=compute_multilabel_metrics,
            threshold=tuned_threshold
        )
        
    print("\n--- Test results at default threshold (0.5) ---")
    log_per_label_metrics(test_result_default.metrics, CHEXPERT_LABELS_14)
    print("\n--- Test results at val-tuned thresholds ---")
    log_per_label_metrics(test_result_tuned.metrics, CHEXPERT_LABELS_14)
    

if __name__ == "__main__":
    main()    