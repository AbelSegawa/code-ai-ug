
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import CodeAI
from model.tokenizer import encode, decode


def main():
    checkpoint_path = ROOT / "checkpoints" / "code_ai_ug.pt"

    if not checkpoint_path.exists():
        print("Trained model not found.")
        print("Download the checkpoint from GitHub Actions first.")
        return

    device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    vocab = checkpoint["vocab"]
    config = checkpoint["config"]

    stoi = {char: index for index, char in enumerate(vocab)}
    itos = {index: char for index, char in enumerate(vocab)}

    model = CodeAI(len(vocab), config).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nCODE AI UG")
    print("Created by Abel Ssegawa")
    print("Type exit to close.\n")

    history = ""

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        prompt = (
            history
            + "User: "
            + question
            + "\nCODE AI UG:"
        )

        tokens = encode(prompt, stoi)

        idx = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            generated = model.generate(
                idx,
                max_new_tokens=200
            )

        output = decode(
            generated[0].tolist(),
            itos
        )

        answer = output[len(prompt):]

        if "\nUser:" in answer:
            answer = answer.split("\nUser:", 1)[0]

        answer = answer.strip()

        print("\nCODE AI UG:", answer, "\n")

        history += (
            "User: "
            + question
            + "\nCODE AI UG: "
            + answer
            + "\n\n"
        )


if __name__ == "__main__":
    main()
