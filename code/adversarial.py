import math
import subprocess
from networks import get_network
from utils.loading import parse_spec
from verifier import analyze
import torch

DEVICE = "cpu"

PRELIMINARY_TEST_CASES = True

""" Please adjust TIMEOUT here """
TIMEOUT = 20

""" Please adjust name here """
name = 'Stuart'

""" Adversarial constants """
EPS_STEP_DECIMAlS = 5

EPS_STEP = math.pow(10, -EPS_STEP_DECIMAlS) # Step size 1 / (10^decimals)
MAX_STEPS = 10e9 # Will stop after approx (10^decimals) steps


torch.set_default_dtype(torch.float32)


if name == 'Angeline':
    project_path = 'c:\\Users\\angel\\Desktop\\ETH\\Master\\3. Semester\\Reliable and Trustworthy AI\\rtai-project-26\\'
    system = 'win'
elif name == 'Stuart':
    project_path = '/Users/stuartheeb/Documents/ETH/HS23/RTAI/rtai-project-26/'
    system = 'mac'
elif name == 'Ioana':
    project_path = 'TODO'
    system = 'mac'

print(name, "is working on a", system)

if PRELIMINARY_TEST_CASES:
    print("Running test cases from preliminary evaluation\n")
    test_cases_folder = "preliminary_evaluation_test_cases"
else:
    print("Running test cases from skeleton\n")
    test_cases_folder = "test_cases"

if system == 'win':
    gt_path = project_path + test_cases_folder +'\\gt.txt'
elif system == 'mac':
    gt_path = project_path + test_cases_folder + '/gt.txt'

unsoundness = False

with open(gt_path, "r") as f:
    test_cases = f.readlines()
    for tc in test_cases:
        model_name = tc.split(',')[0]
        image_path = tc.split(',')[1]
        gt = tc.split(',')[2].rstrip("\n")
        assert gt == "verified" or gt == "not verified"

        print(model_name, ",", image_path, ",", gt)

        spec = test_cases_folder + "/" + model_name + "/" + image_path

        true_label, dataset, image, eps = parse_spec(spec)

        net = get_network(model_name, dataset, f"models/{dataset}_{model_name}.pt").to(DEVICE)

        image = image.to(DEVICE).unsqueeze(0)

        out = net(image)

        pred_label = out.max(dim=1)[1].item()
        assert pred_label == true_label

        first1_eps = None
        first2_eps = None
        i = 1
        curr_eps = 0
        while i <= MAX_STEPS:
            curr_eps += EPS_STEP
            curr_eps = round(curr_eps, EPS_STEP_DECIMAlS)
            if(curr_eps > 1.0):
                print("No longer perturbing")
                break

            perturbed1 = (image + curr_eps * torch.ones_like(image)).clamp(0, 1)
            assert (perturbed1 >= 0.0).all() and (perturbed1 <= 1.0).all()
            perturbed2 = (image - curr_eps * torch.ones_like(image)).clamp(0, 1)
            assert (perturbed2 >= 0.0).all() and (perturbed2 <= 1.0).all()

            # PLUS PERTURBATION
            perb_out1 = net(perturbed1)
            pred1 = perb_out1.max(dim=1)[1].item()
            if pred1 != true_label:
                if first1_eps is None:
                    first1_eps = curr_eps
                    print("> EPS = +", first1_eps)
            else:
                if first1_eps is not None:
                    print("Found match again for EPS = +", curr_eps)

            # MINUS PERTURBATION
            perb_out2 = net(perturbed2)
            pred2 = perb_out2.max(dim=1)[1].item()
            if pred2 != true_label:
                if first2_eps is None:
                    #first2_eps = -curr_eps
                    first2_eps = curr_eps
                    print("> EPS = -", first2_eps)
            else:
                if first2_eps is not None:
                    print("Found match again for EPS = -", curr_eps)

            i += 1


        # TODO Run using subprocess with the timeout, for this need to change the image_path to include the new epsilon
        if first1_eps is not None:
            if analyze(net, image, first1_eps, true_label):
                unsoundness = True
                print("UNSOUND for EPS =", first1_eps)

        # TODO same here
        if first2_eps is not None:
            if analyze(net, image, first2_eps, true_label):
                unsoundness = True
                print("UNSOUND for EPS =", first2_eps)

if unsoundness:
    print("UNSOUNDNESS!")
else:
    print("No unsoundness detected.")