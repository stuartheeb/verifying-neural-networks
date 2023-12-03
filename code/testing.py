import torch
# import torch.nn as nn

from networks import get_network

from verifier import analyze

DEVICE = "cpu"

""" Please adjust name here """
name = 'Stuart'

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


def test_case(model_name, image_path):
    path = 'test_cases/' + model_name + '/' + image_path
    eps = float(".".join(path.split("/")[-1].split("_")[-1].split(".")[:2]))
    dataset = path.split("/")[-1].split("_")[1]
    shape = (1, 1, 28, 28) if "mnist" in dataset else (1, 3, 32, 32)

    if system == 'win':
        full_path = project_path + 'test_cases\\' + model_name + '\\' + image_path
        pt_path = project_path + "models\\" + dataset + "_" + model_name + ".pt"
    elif system == 'mac':
        full_path = project_path + 'test_cases/' + model_name + '/' + image_path
        pt_path = project_path + "models/" + dataset + "_" + model_name + ".pt"

    with open(full_path, "r") as f:
        # First line is the label
        label = int(f.readline().strip())
        # Second line is the image
        image = [float(x) for x in f.readline().strip().split(",")]
    image = torch.tensor(image).reshape(shape)

    net = get_network(model_name, dataset, pt_path).to(DEVICE)
    image = image.to(DEVICE)

    return net, image, eps, label


if system == 'win':
    gt_path = project_path + 'test_cases\\gt.txt'
elif system == 'mac':
    gt_path = project_path + 'test_cases/gt.txt'

total = 0
exact = 0
imprecise = 0
potentially_unsound = 0

with open(gt_path, "r") as f:
    test_cases = f.readlines()
    for tc in test_cases:
        model_name = tc.split(',')[0]
        image_path = tc.split(',')[1]
        gt = tc.split(',')[2].rstrip("\n")

        print("Test case:", model_name, ',', image_path, ',', gt)

        net, image, eps, label = test_case(model_name, image_path)
        if analyze(net, image, eps, label):
            result = "verified"
        else:
            result = "not verified"

        print("Result:", result)

        total += 1
        if result == "verified" and gt == "not verified":
            potentially_unsound += 1
            print("> POTENTIALLY UNSOUND: verified when ground truth couldn't")
        elif result == "not verified" and gt == "verified":
            imprecise += 1
            # print("> imprecise: couldn't verify when ground truth could")
        else:
            exact += 1
            # print("> exact match")

        # print('\n')

print("Total test cases run:", total)
print("Exact matches:", exact)
print("Imprecise:", imprecise)
print("Potentially unsound:", potentially_unsound)