from prepare_data import load_and_prepare_data
from infer import load_model_for_inference, generate


def evaluate(n_samples=5, use_adapter=True):
    _, eval_data = load_and_prepare_data()
    model, tokenizer = load_model_for_inference(use_adapter=use_adapter)

    samples = eval_data.select(range(min(n_samples, len(eval_data))))

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        expected = sample["output"]

        response = generate(model, tokenizer, instruction, max_new_tokens=200)

        print(f"\n{'='*60}")
        print(f"Example {i+1}")
        print(f"{'='*60}")
        print(f"\nInstruction:\n{instruction}")
        print(f"\nExpected:\n{expected}")
        print(f"\nModel Output:\n{response}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--no_adapter", action="store_true")
    args = parser.parse_args()
    evaluate(n_samples=args.n, use_adapter=not args.no_adapter)
