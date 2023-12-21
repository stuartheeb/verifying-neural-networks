import subprocess
import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

DEVICE = "cpu"

PRELIMINARY_TEST_CASES = True

""" Please adjust TIMEOUT here """
TIMEOUT = 20

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

if PRELIMINARY_TEST_CASES:
    print("Running test cases from preliminary evaluation\n")
    test_cases_folder = "preliminary_evaluation_test_cases"
    server_runtimes = pd.read_csv("preliminary_evaluation.csv")
else:
    print("Running test cases from skeleton\n")
    test_cases_folder = "test_cases"

if system == 'win':
    gt_path = project_path + test_cases_folder + '\\gt.txt'
elif system == 'mac':
    gt_path = project_path + test_cases_folder + '/gt.txt'

for optimizer in ["adamw", "adam", "sgd", "adagrad", "adadelta", "rmsprop"]:
    total_local_runtimes = []
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print("\nOptimizer:", optimizer)
        print("LR = ", lr)
        total = 0
        exact = 0
        imprecise = 0
        unsound = 0
        timeouts_exact = 0
        timeouts_imprecise = 0
        points = 0
        total_points = 0

        runtimes = []
        verified_runtimes = []
        with open(gt_path, "r") as f:
            test_cases = f.readlines()
            for tc in test_cases:
                model_name = tc.split(',')[0]
                image_path = tc.split(',')[1]
                gt = tc.split(',')[2].rstrip("\n")
                assert gt == "verified" or gt == "not verified"

                if gt == "not verified":  # For optimizing LR just use verifiable test cases
                    continue

                if points != total_points: # Only want to compare LRs which achieve full points (early stopping mechanism for quicker execution)
                    continue

                print("Test case:", model_name, ',', image_path, ',', gt)
                timed_out = False
                try:
                    spec = test_cases_folder + "/" + model_name + "/" + image_path
                    # cmd = "python code/verifier.py --net "+model_name+" --spec test_cases/"+model_name+"/"+image_path

                    start = time.time()
                    if TIMEOUT is None:
                        r = subprocess.run(
                            ["python", "code/verifier_lr.py", "--net", model_name, "--spec", spec, "--lr", str(lr),
                             "--optimizer", optimizer],
                            capture_output=True)
                    else:
                        r = subprocess.run(
                            ["python", "code/verifier_lr.py", "--net", model_name, "--spec", spec, "--lr", str(lr),
                             "--optimizer", optimizer], timeout=TIMEOUT,
                            capture_output=True)
                    end = time.time()
                    result = r.stdout.decode("utf-8").strip("\n")
                    local_runtime = round(end - start, 3)
                    if gt == "verified":
                        verified_runtimes.append(local_runtime)
                    print("Local runtime =", local_runtime, "s")

                except subprocess.TimeoutExpired as e:
                    print("TIMEOUT")
                    timed_out = True
                    local_runtime = TIMEOUT
                    result = "not verified"

                if result == "":
                    print("RESULT COULD NOT BE PARSED")
                    result = "not verified"
                assert result == "verified" or result == "not verified"

                if PRELIMINARY_TEST_CASES:
                    index = model_name + "_" + image_path.strip(".txt")
                    server_runtime = \
                        (server_runtimes.loc[server_runtimes['net'] == model_name].loc[
                            server_runtimes['test case'] == image_path])[
                            "your runtime"].values[0]
                    speedup = round(server_runtime / local_runtime, 3)
                    print("Server runtime =", server_runtime, "s,", speedup, "x")
                    if not timed_out and server_runtime < 60:
                        runtimes.append([local_runtime, server_runtime, speedup])
                else:
                    if not timed_out:
                        runtimes.append([local_runtime, np.NaN, np.NaN])

                print("Result:", result)

                total += 1
                if result == "verified" and gt == "not verified":
                    unsound += 1
                    points -= 3
                    print("> UNSOUND: verified when ground truth couldn't")
                elif result == "not verified" and gt == "verified":
                    imprecise += 1
                    if timed_out:
                        timeouts_imprecise += 1
                    else:
                        print("> imprecise: couldn't verify when ground truth could")
                else:
                    exact += 1
                    if gt == "verified":
                        points += 1
                    if timed_out:
                        timeouts_exact += 1
                if gt == "verified":
                    total_points += 1
                print("")

        print("Total test cases run:", total)
        print("Exact matches:", exact, "(of which", timeouts_exact, "are timeouts)")
        print("Imprecise:", imprecise, "(of which", timeouts_imprecise, "are timeouts)")
        print("Timeouts:", timeouts_exact + timeouts_imprecise)
        print("Unsound:", unsound)
        print("Total points:", points, "/", total_points)

        if len(verified_runtimes)>0:
            verified_runtimes = np.array(verified_runtimes)
            if points == total_points:
                total_local_runtimes.append([lr, verified_runtimes.sum()])

        if len(runtimes)>0:
            runtimes = np.array(runtimes)
            local_runtimes = runtimes[:, 0]

            print("\nTotal local runtime (incl. timeouts) =", local_runtimes.sum(), "s +",
                  (timeouts_exact + timeouts_imprecise), "*", TIMEOUT, "s =",
                  round(local_runtimes.sum() + TIMEOUT * (timeouts_exact + timeouts_imprecise), 3), "s")
            print("Max. local runtime =", local_runtimes.max(), "s")

            print("Local runtimes (excl. timeouts) =", runtimes[:, 0])

            if PRELIMINARY_TEST_CASES:
                print("\n(The following only makes sense if using the same parameters as during preliminary evaluation)")
                print("Speedup on local machine compared to server (timeouts on either side are not considered)")
                speedups = runtimes[:, 2]
                print("min =", speedups.min())
                print("max =", speedups.max())
                print("mean =", speedups.mean())
                print("std =", speedups.std())

    if len(total_local_runtimes)>0: # None of the LRs reached maximum points
        total_local_runtimes = np.array(total_local_runtimes)
        print(total_local_runtimes)
        fig, ax = plt.subplots()
        ax.semilogx(total_local_runtimes[:, 0], total_local_runtimes[:, 1], "ro")
        ax.set(title=str(optimizer) + ': LR vs runtime')
        ax.grid()
        plt.ylabel("runtime [s]")
        plt.xlabel("learning rate")
        plt.savefig(str(test_cases_folder) + "/plots/" + str(optimizer) + "_lr_plot.pdf")