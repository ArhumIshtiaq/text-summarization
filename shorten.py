import torch
from summarizer import TransformerSummarizer
from transformers.utils.dummy_pt_objects import XLNetLMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, XLNetLMHeadModel, GPT2Tokenizer,GPT2LMHeadModel

import warnings
warnings.filterwarnings("ignore")

def readfile(filename):
    txt = ""
    f = open(filename)
    for line in f:
        txt += line
    f.close()
    return txt

def T5():
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    inputtxt = readfile("vax.txt")
    preprocess_text = inputtxt.strip().replace("\n","")
    t5_prepared_text = "summarize: "+ preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt")
    summary_ids = model.generate(tokenized_text, do_sample=True, num_beams=2, min_length=100, top_p=0.95, num_return_sequences=5, early_stopping=True)
    print("ORIGINAL INPUT:")
    print(inputtxt)
    print("SUMMARY:")
    print(tokenizer.decode(summary_ids[0], skip_special_tokens=False))


def XLNet():
    xlnet = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    inputtxt = readfile("vax.txt")
    full = ''.join(xlnet(inputtxt, min_length=100))
    print("ORIGINAL INPUT:")
    print(inputtxt.strip(),end="\n\n")
    print("SUMMARY:")
    print(full)


def gpt2():
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    model=GPT2LMHeadModel.from_pretrained('gpt2')
    inputtxt = readfile("vax.txt")
    preprocess_text = inputtxt.strip().replace("\n","")
    inputs=tokenizer.batch_encode_plus([preprocess_text],return_tensors='pt')
    summary_ids=model.generate(inputs['input_ids'],early_stopping=True,max_length=200)
    GPT_summary=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    print(GPT_summary)


i = int(input("Choose 1 for T5, 2 for XLNet, 3 for GPT-2: "))

if (i == 1):
    T5()
elif (i == 2):
    XLNet()
elif (i == 3):
    gpt2()
else:
    print("Error: Invalid input.")