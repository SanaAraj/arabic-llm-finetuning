import argparse


def main():
    parser = argparse.ArgumentParser(description="Arabic LLM Fine-tuning Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare and preview dataset")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--max_steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--prompt", type=str, help="Prompt for inference")
    parser.add_argument("--n_eval", type=int, default=5, help="Number of eval samples")
    args = parser.parse_args()

    if args.prepare:
        from prepare_data import preview_data
        preview_data()

    elif args.train:
        from train import train
        train(max_steps=args.max_steps, batch_size=args.batch_size, learning_rate=args.lr)

    elif args.infer:
        if not args.prompt:
            parser.error("--infer requires --prompt")
        from infer import run_inference
        run_inference(args.prompt)

    elif args.eval:
        from evaluate import evaluate
        evaluate(n_samples=args.n_eval)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
