import os
import sys
import glob
import types
import argparse
import textwrap
import subprocess
import pickle
import concurrent.futures
import shutil
import pandas as pd
from Bio import SeqIO


def find_modeller_executable():
    for name in ("mod10.8", "mod9.23", "mod9.21", "mod9.19"):
        exe = shutil.which(name)
        if exe:
            return exe

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates = sorted(glob.glob(os.path.join(conda_prefix, "bin", "mod*")))
        candidates = [path for path in candidates if os.access(path, os.X_OK)]
        if candidates:
            return candidates[-1]

    raise RuntimeError(
        "Modeller executable not found. Install Modeller and ensure a 'mod9.xx' "
        "binary is on PATH, or pass --modeller explicitly."
    )


def register_sklearn_pickle_compat():
    import sklearn.linear_model as lm
    from sklearn.preprocessing import StandardScaler

    if "sklearn.linear_model.logistic" not in sys.modules:
        logistic_module = types.ModuleType("sklearn.linear_model.logistic")
        logistic_module.LogisticRegression = lm.LogisticRegression
        sys.modules["sklearn.linear_model.logistic"] = logistic_module

    if "sklearn.preprocessing.data" not in sys.modules:
        preprocessing_module = types.ModuleType("sklearn.preprocessing.data")
        preprocessing_module.StandardScaler = StandardScaler
        sys.modules["sklearn.preprocessing.data"] = preprocessing_module


def safe_link(src, dst):
    if not os.path.exists(dst):
        os.link(src, dst)


def inParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", required=True, help="Input fasta file")
    parser.add_argument("-o", "--output", default='output' , help = "Output directory path")
    parser.add_argument(
        "--modeller",
        default=None,
        help="Modeller executable (default: auto-detect from PATH)",
    )
    args = parser.parse_args()

    #if args.fasta:
    #    parser.error("Wrong operation")
    
    return(args)


def temlateSeq(fasta, template):

    record = SeqIO.parse(fasta, 'fasta')  

    for i in record:
        if template == i.name:
            return(str(i.seq))


def infile(sequence, structure, fasta):
    
    t_seq = temlateSeq(fasta, structure)
    
    print(t_seq)
    f = open('alignment.pir', 'w')

    print(">P1;template", file = f)
    print("Structure:%s:FIRST:A:LAST:L::-1.00:-1.00:" % structure, file = f)
    for i in range(1,12):
        print(textwrap.fill(t_seq, width = 80) + '/', file = f)
    print(textwrap.fill(t_seq, width = 80) + '*\n', file = f)

    print(">P1;sequence", file = f)
    print("sequence:target::::::-1.00:-1.00:", file = f)
    #print(textwrap.fill(t_seq, width = 80), file = f)
    for i in range(1,12):
        print(textwrap.fill(sequence, width = 80) + '/', file = f)
    print(textwrap.fill(sequence, width = 80) + '*\n', file = f)
    
    f.close()


def runScript(script, quiet=False):
    kwargs = {"shell": True, "stdin": subprocess.DEVNULL}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    subprocess.run(script, **kwargs)


def modeller_log_has_results(log_path):
    if not os.path.exists(log_path):
        return False

    with open(log_path) as log_file:
        for line in log_file:
            if line.startswith("sequence.") and ".pdb" in line:
                return True

    return False


def model(sequence, output_path, modeller_exe, project_root, template_fasta_path):
    dir_path = os.path.join(output_path, sequence)
    os.makedirs(dir_path, exist_ok=True)

    for template in ['class1', 'class2', 'class4', 'class5', 'class6', 'class7', 'class8']:
        models_path = os.path.join(dir_path, template)
        os.makedirs(models_path, exist_ok=True)

        log_path = os.path.join(models_path, "runMod.log")
        if modeller_log_has_results(log_path):
            continue

        template_path = os.path.join(project_root, "templates", template + ".pdb")
        runmod_script = os.path.join(project_root, "scripts", "runMod.py")
        safe_link(template_path, os.path.join(models_path, template + ".pdb"))
        safe_link(runmod_script, os.path.join(models_path, "runMod.py"))

        previous_dir = os.getcwd()
        os.chdir(models_path)

        try:
            infile(sequence, template, template_fasta_path)
            runScript(f"{modeller_exe} runMod.py", quiet=True)
        finally:
            os.chdir(previous_dir)


if __name__ == '__main__':
  
    args = inParser()

    input_path = args.fasta
    output_path = args.output
    project_root = os.path.abspath(os.path.dirname(__file__))
           
    # Modeling
    os.makedirs(output_path, exist_ok=True)

    modeller_exe = args.modeller or find_modeller_executable()

    fasta = SeqIO.parse(input_path,'fasta')

    template_fasta = os.path.join(project_root, "templates.fasta")

    proteins = []
    prot_names = []
    for record in fasta:
        sequences = str(record.seq)
        if len(sequences) > 6:
            subseqs = [sequences[i:i+6] for i in range(len(sequences)-5)]
            tmp = []
            sequences = [tmp.append(subseqs[i]) for i in range(len(subseqs))]
            sequences = tmp
        else:
            sequences = [sequences]
            print(sequences)
        
        prot_names.append(record.id)
        proteins.append(sequences)
        
        #Add multiprocessing

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for s in sequences:
                executor.submit(
                    model, s, output_path, modeller_exe, project_root, template_fasta
                )
        
    # Scoring
    os.chdir(output_path)

    scripts_dir = os.path.join(project_root, "scripts")
    safe_link(os.path.join(scripts_dir, "data.sh"), "data.sh")
    safe_link(os.path.join(scripts_dir, "score2.py"), "score2.py")
    
    runScript("bash data.sh > results.csv")
    runScript(f"{sys.executable} score2.py")
    
    # Analysis
    register_sklearn_pickle_compat()

    models_dir = os.path.join(project_root, "trained_models", "waltz")
    with open(os.path.join(models_dir, "LR"), "rb") as infile:
        clf = pickle.load(infile)

    with open(os.path.join(models_dir, "scaler"), "rb") as infile:
        scaler = pickle.load(infile)
    
    modeller = pd.read_csv("results.csv")
    rosetta = pd.read_csv("pyrosetta.csv")

    data = pd.merge(modeller, rosetta, on=['seq','class','model'])
    if data.empty:
        raise RuntimeError(
            "No Modeller results found. Verify Modeller is installed, licensed, "
            "and that runMod.log files were created under the output directory."
        )

    min_e = data.loc[data.groupby('seq')['dope'].idxmin()]

    min_inputs = min_e.values[:,3:]
    inputs = scaler.transform(min_inputs)
    seq = min_e.values[:,0]

    prediction = clf.predict(inputs)

    results = pd.DataFrame(seq, columns=['seq'])
    results['path'] = prediction

    results.to_csv('hexapeptides.csv', index=False)

    pred_d = {seq[i]:prediction[i] for i in range(len(seq))}
    print("Amylodogenic fragments:")

    for p in range(len(prot_names)):
        print(prot_names[p])
        for hexapeptide in proteins[p]:
            if pred_d[hexapeptide] == 1:
                print(hexapeptide)
