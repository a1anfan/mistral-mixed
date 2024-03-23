import argparse

from ngrams.make_corpus import *

def merge_corpuses(ckpt_path, suffix="cml"):
  prefixes = ["b_d", "t_d", "fo_d", "fi_d", "si_d"]
  for prefix in prefixes: 
    corpus = None 
    for filepath in glob.glob(f"{ckpt_path}/{prefix}*"):
        with open(filepath, "rb") as f:
            current = pickle.load(f)
            if corpus is None:
                corpus = current
            else:
                corpus = merge_corpus_helper(corpus, current)
        os.remove(filepath)
    with open(f"{ckpt_path}/{prefix}_{suffix}.pkl", "wb") as f: 
      pickle.dump(corpus, f)
  
def merge_counts(ckpt_path, suffix="cml"):
  prefixes = ["b_ct", "t_ct", "fo_ct", "fi_ct", "si_ct"]
  for prefix in prefixes: 
    counts = None 
    for filepath in glob.glob(f"{ckpt_path}/{prefix}*"):
      with open(filepath, "rb") as f:
        current = pickle.load(f)
        if counts is None:
          counts = current
        else:
          counts = merge_counts_helper(counts, current)
      os.remove(filepath)
    with open(f"{ckpt_path}/{prefix}_{suffix}.pkl", "wb") as f: 
      pickle.dump(counts, f)
      
parser = argparse.ArgumentParser()
parser.add_argument("ckpt_path", type=str)
# parser.add_argument("start_doc", type=str)
# parser.add_argument("end_doc", type=str)
args = parser.parse_args()
# merge_corpuses(ckpt_path=args.ckpt_path, suffix=f"final")
merge_counts(ckpt_path=args.ckpt_path, suffix=f"final")