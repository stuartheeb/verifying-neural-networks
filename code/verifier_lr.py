import argparse
import torch

from networks import get_network
from utils.loading import parse_spec

from DeepPoly import DeepPolySequential, DeepPolyVerifier, DeepPolyLoss, DeepPolyConstraints

DEVICE = "cpu"

NUM_EPOCHS = 100000000


def analyze(
        net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int, lr: float, optimizer: str
) -> bool:
    dp_sequential = DeepPolySequential(net, inputs)
    dp_verifier = DeepPolyVerifier(true_label)
    dp_loss = DeepPolyLoss()
    dp_constraints = DeepPolyConstraints.constraints_from_eps(inputs, eps, clipper=(0.0, 1.0))

    if len(list(dp_sequential.parameters())) == 0:
        verification = dp_verifier(dp_sequential(dp_constraints))
        if (verification.lbounds > 0.0).all():
            return True
        else:
            return False

    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(dp_sequential.parameters(), lr=lr)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(dp_sequential.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(dp_sequential.parameters(), lr=lr)
    elif optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(dp_sequential.parameters(), lr=lr)
    elif optimizer == "adadelta":
        optimizer = torch.optim.Adagrad(dp_sequential.parameters(), lr=lr)
    elif optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(dp_sequential.parameters(), lr=lr)
    else:  # Optimizer
        optimizer = None
    assert optimizer is not None

    # for _ in range(NUM_EPOCHS):
    while True:
        optimizer.zero_grad()
        verification = dp_verifier(dp_sequential(dp_constraints))
        if (verification.lbounds > 0.0).all():
            return True
        loss = dp_loss(verification)
        loss.backward()
        optimizer.step()

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, required=True, help="Optimizer.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE).unsqueeze(0)
    out = net(image)

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    lr = float(args.lr)
    if analyze(net, image, eps, true_label, lr, args.optimizer):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
