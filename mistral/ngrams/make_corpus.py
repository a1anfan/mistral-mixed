import multiprocessing
import argparse
import os
import pickle
import glob
from datasets import load_dataset
from tqdm import tqdm
from transformers import LlamaTokenizer
from loguru import logger


def create_corpuses(
    start_doc,
    end_doc,
    dataset, 
    tokenizer, 
    train_bigram: bool, 
    train_trigram: bool,
    train_fourgram: bool,
    train_fivegram: bool,
    train_sixgram: bool,
    train_sevengram: bool
):
    bigram_corpus = {}
    trigram_corpus = {}
    fourgram_corpus = {}
    fivegram_corpus = {}
    sixgram_corpus = {}
    sevengram_corpus = {}

    bigram_corpus_counts = {}
    trigram_corpus_counts = {}
    fourgram_corpus_counts = {}
    fivegram_corpus_counts = {}
    sixgram_corpus_counts = {}
    sevengram_corpus_counts = {}
    
    iterations = end_doc - start_doc
    for i in tqdm(range(iterations)):
      t = dataset[start_doc + i]["text"]
      encoded_text = tokenizer.encode(t)
      for start_idx in range(1, len(encoded_text)): # count from first real to eos
        pOne = encoded_text[start_idx-1] if start_idx >= 1 else None
        pTwo = encoded_text[start_idx-2] if start_idx >= 2 else None
        pThree = encoded_text[start_idx-3] if start_idx >= 3 else None
        pFour = encoded_text[start_idx-4] if start_idx >= 4 else None
        pFive = encoded_text[start_idx-5] if start_idx >= 5 else None
        pSix = encoded_text[start_idx-6] if start_idx >= 6 else None
        
        token = encoded_text[start_idx]
        # bigram
        if train_bigram and start_idx >= 1:
          prior = pOne
          if prior not in bigram_corpus:
            bigram_corpus[prior] = {}
            bigram_corpus_counts[prior] = 0
          bigram_corpus[prior][token] = bigram_corpus[prior].get(token, 0) + 1
          bigram_corpus_counts[prior] += 1
        # trigram 
        if train_trigram and start_idx >= 2:
          prior = (pTwo, pOne)
          if prior not in trigram_corpus:
            trigram_corpus[prior] = {}
            trigram_corpus_counts[prior] = 0
          trigram_corpus[prior][token] = trigram_corpus[prior].get(token, 0) + 1
          trigram_corpus_counts[prior] += 1
        # fourgram
        if train_fourgram and start_idx >= 3: 
          prior = (pThree, pTwo, pOne)
          if prior not in fourgram_corpus:
            fourgram_corpus[prior] = {}
            fourgram_corpus_counts[prior] = 0
          fourgram_corpus[prior][token] = fourgram_corpus[prior].get(token, 0) + 1
          fourgram_corpus_counts[prior] += 1     
        # fivegram
        if train_fivegram and start_idx >= 4:
          prior = (pFour, pThree, pTwo, pOne)
          if prior not in fivegram_corpus:
            fivegram_corpus[prior] = {}
            fivegram_corpus_counts[prior] = 0
          fivegram_corpus[prior][token] = fivegram_corpus[prior].get(token, 0) + 1
          fivegram_corpus_counts[prior] += 1            
        # sixgram
        if train_sixgram and start_idx >= 5:
          prior = (pFive, pFour, pThree, pTwo, pOne)
          if prior not in sixgram_corpus:
            sixgram_corpus[prior] = {}
            sixgram_corpus_counts[prior] = 0
          sixgram_corpus[prior][token] = sixgram_corpus[prior].get(token, 0) + 1
          sixgram_corpus_counts[prior] += 1     
        # sevengram
        if train_sevengram and start_idx >= 6:
          prior = (pSix, pFive, pFour, pThree, pTwo, pOne)
          if prior not in sevengram_corpus:
            sevengram_corpus[prior] = {}
            sevengram_corpus_counts[prior] = 0
          sevengram_corpus[prior][token] = sevengram_corpus[prior].get(token, 0) + 1
          sevengram_corpus_counts[prior] += 1  
    save_corpus(bigram_corpus, trigram_corpus, fourgram_corpus, fivegram_corpus, sixgram_corpus, sevengram_corpus, start_doc, end_doc)
    save_counts(bigram_corpus_counts, trigram_corpus_counts, fourgram_corpus_counts, fivegram_corpus_counts, sixgram_corpus_counts, sevengram_corpus_counts, start_doc, end_doc)    
    # return (bigram_corpus, bigram_corpus_counts), (trigram_corpus, trigram_corpus_counts), (fourgram_corpus, fourgram_corpus_counts), (fivegram_corpus, fivegram_corpus_counts), (sixgram_corpus, sixgram_corpus_counts), (sevengram_corpus, sevengram_corpus_counts)

def merge_corpus_helper(c1, c2):
  for prior in c2:
    # if share prior
    if prior in c1:
      c1_prior = c1[prior]
      c2_prior = c2[prior]
      for token in c2_prior:
        # if share token
        if token in c1_prior:
          c1_prior[token] += c2_prior[token]
        # else just use c2's
        else:
          c1_prior[token] = c2_prior[token]
    else:
      # else just use c2's
      c1[prior] = c2[prior]
  return c1

def merge_counts_helper(c1, c2):
  for prior in c2:
    if prior in c1:
      c1[prior] += c2[prior]
    else:
      c1[prior] = c2[prior]
  return c1

def save_corpus(b_d, t_d, fo_d, fi_d, si_d, se_d, start_doc, end_doc):
  save_dir = "/gscratch/raivn/ethans/llms/llama-mixed/ckpts"
  prefixes = ["b_d", "t_d", "fo_d", "fi_d", "si_d", "se_d"]
  for p, corpus in zip(prefixes, [b_d, t_d, fo_d, fi_d, si_d, se_d]):
    with open(f"{save_dir}/{p}{start_doc}-{end_doc}.pkl", "wb") as f:
      pickle.dump(corpus, f)

def save_counts(b_ct, t_ct, fo_ct, fi_ct, si_ct, se_ct, start_doc, end_doc):
  save_dir = "/gscratch/raivn/ethans/llms/llama-mixed/ckpts"
  prefixes = ["b_ct", "t_ct", "fo_ct", "fi_ct", "si_ct", "se_ct"]
  for p, corpus in zip(prefixes, [b_ct, t_ct, fo_ct, fi_ct, si_ct, se_ct]):
    with open(f"{save_dir}/{p}{start_doc}-{end_doc}.pkl", "wb") as f:
      pickle.dump(corpus, f)
      
def merge_corpuses(ckpt_path):
  prefixes = ["b_d", "t_d", "fo_d", "fi_d", "si_d", "se_d"]
  for prefix in prefixes: 
    if os.path.exists(f"{ckpt_path}/{prefix}_final.pkl"):
      os.remove(f"{ckpt_path}/{prefix}_final.pkl")
    
    corpus = None 
    for filepath in glob.glob(f"{ckpt_path}/{prefix}*"):
      with open(filepath, "rb") as f:
        current = pickle.load(f)
        if corpus is None:
          corpus = current
        else:
          corpus = merge_corpus_helper(corpus, current)
      os.remove(filepath)
    with open(f"{ckpt_path}/{prefix}_final.pkl", "wb") as f: 
      pickle.dump(corpus, f)
  
def merge_counts(ckpt_path):
  prefixes = ["b_ct", "t_ct", "fo_ct", "fi_ct", "si_ct", "se_ct"]
  for prefix in prefixes: 
    if os.path.exists(f"{ckpt_path}/{prefix}_final.pkl"):
      os.remove(f"{ckpt_path}/{prefix}_final.pkl")
      
    counts = None 
    for filepath in glob.glob(f"{ckpt_path}/{prefix}*"):
      with open(filepath, "rb") as f:
        current = pickle.load(f)
        if counts is None:
          counts = current
        else:
          counts = merge_counts_helper(counts, current)
      os.remove(filepath)
    with open(f"{ckpt_path}/{prefix}_final.pkl", "wb") as f: 
      pickle.dump(counts, f)
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("start_doc", type=str)
  parser.add_argument("end_doc", type=str)
  parser.add_argument("c", type=int)
  args = parser.parse_args()
  start_doc_ovr = int(args.start_doc)
  end_doc_ovr = int(args.end_doc)
  n_cores = args.c
  logger.info(f"{start_doc_ovr} {end_doc_ovr} {n_cores}")
  
  # setup
  rpj = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir="../datasets/")["train"].shuffle(seed=42)
  tokenizer = LlamaTokenizer.from_pretrained("./7B_HF", add_bos_token=False)

  # train
  train_bigram = True # not os.path.exists(f"{data_dir}/bigram.pt")
  train_trigram = True #not os.path.exists(f"{data_dir}/trigram.pt")
  train_fourgram = True
  train_fivegram = True
  train_sixgram = True
  train_sevengram = True

  num_processes = n_cores
  total_docs = end_doc_ovr - start_doc_ovr
  docs_per_c = (total_docs) // num_processes
  processes = []
  for core in range(n_cores):
    start_doc = core * docs_per_c # relative start doc 
    end_doc = (core + 1) * docs_per_c if core < n_cores - 1 else total_docs # relative end doc
    logger.info(f"Starting core {core} from {start_doc} to {end_doc}")
    process = multiprocessing.Process(target=create_corpuses, args=(start_doc_ovr + start_doc, start_doc_ovr + end_doc, rpj, tokenizer, train_bigram, train_trigram, train_fourgram, train_fivegram, train_sixgram, train_sevengram))
    processes.append(process)
    process.start()
  for process in processes:
    process.join()
  logger.info("Finished Saving")
  
  logger.info("Merging...")
  merge_corpuses("./ckpts")
  merge_counts("./ckpts")
      
