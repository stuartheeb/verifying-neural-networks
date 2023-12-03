import subprocess

DEVICE = "cpu"

""" Please adjust TIMEOUT here """
TIMEOUT = None

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

print(name, "is working on a", system, "\n")


if system == 'win':
    gt_path = project_path + 'test_cases\\gt.txt'
elif system == 'mac':
    gt_path = project_path + 'test_cases/gt.txt'

total = 0
exact = 0
imprecise = 0
potentially_unsound = 0
timeout = 0

with open(gt_path, "r") as f:
    test_cases = f.readlines()
    for tc in test_cases:
        model_name = tc.split(',')[0]
        image_path = tc.split(',')[1]
        gt = tc.split(',')[2].rstrip("\n")

        print("Test case:", model_name, ',', image_path, ',', gt)

        if model_name in "conv":
            continue

        try:
            spec = "test_cases/" + model_name + "/" + image_path
            # command: "python code/verifier.py --net "+model_name+" --spec test_cases/"+model_name+"/"+image_path
            if TIMEOUT is None:
                r = subprocess.run(["python", "code/verifier.py", "--net", model_name, "--spec", spec], capture_output=True)
            else:
                r = subprocess.run(["python", "code/verifier.py", "--net", model_name, "--spec", spec], timeout=20, capture_output=True)
            result = r.stdout.decode("utf-8").strip("\n")
        except subprocess.TimeoutExpired as e:
            print("TIMEOUT")
            result = "timeout"

        print("Result:", result, "\n")

        total += 1
        if result == "timeout":
            timeout += 1
        elif result == "verified" and gt == "not verified":
            potentially_unsound += 1
            print("> POTENTIALLY UNSOUND: verified when ground truth couldn't")
        elif result == "not verified" and gt == "verified":
            imprecise += 1
            # print("> imprecise: couldn't verify when ground truth could")
        else:
            exact += 1
            # print("> exact match")

print("Total test cases run:", total)
print("Exact matches:", exact)
print("Imprecise:", imprecise)
print("Timeouts:", timeout)
print("Potentially unsound:", potentially_unsound)
