import os

os.system('pip install -q -e .')
os.system('pip uninstall bitsandbytes')
os.system('pip install bitsandbytes-0.45.0-py3-none-manylinux_2_24_x86_64.whl')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
print(torch.cuda.is_available()) 

print(os.system('python -m bitsandbytes'))

import gradio as gr
import io
from contextlib import redirect_stdout
import openai
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaMistralForCausalLM
from llava.eval.run_llava import eval_model

# LLaVa-Med model setup
model_path = "Veda0718/llava-med-v1.5-mistral-7b-finetuned"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

def query_gpt(api_key, llava_med_result, user_question, model="gpt-4o"):
    """
    Queries GPT to generate a detailed and medically accurate response.
    """
    openai.api_key = api_key  # Set API key dynamically
    prompt = f"""
    You are an AI Medical Assistant specializing in radiology, trained to analyze radiology scan findings (e.g., MRI, CT, X-ray) and provide clear, medically accurate explanations. 
    Based on the scan analysis {llava_med_result} and the question {user_question}, provide a concise summary of the radiology findings. Use clear, professional language, and if uncertain, 
    recommend consulting a licensed radiologist or healthcare provider.
    """
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256
    )
    return response.choices[0].message.content

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    with gr.Column(scale=1):
        gr.Markdown("<center><h1>LLaVa-Med</h1></center>")

        with gr.Row():
            api_key_input = gr.Textbox(
                placeholder="Enter OpenAI API Key", 
                label="API Key", 
                type="password",
                scale=3
            )

        with gr.Row():
            image = gr.Image(type="filepath", scale=2)
            question = gr.Textbox(placeholder="Enter a question", label="Question", scale=3)

        with gr.Row():
            answer = gr.Textbox(placeholder="Answer pops up here", label="Answer", scale=1)

        def run_inference(api_key, image, question):
            # Arguments for the model
            args = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "image_file": image,
                "query": question,
                "conv_mode": None,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()

            # Capture the printed output of eval_model
            f = io.StringIO()
            with redirect_stdout(f):
                eval_model(args)
            llava_med_result = f.getvalue()
            print(llava_med_result)

            # Generate more descriptive answer with GPT
            descriptive_answer = query_gpt(api_key, llava_med_result, question)

            return descriptive_answer

        with gr.Row():
            btn = gr.Button("Run Inference", scale=1)

        btn.click(fn=run_inference, inputs=[api_key_input, image, question], outputs=answer)

app.launch(debug=True, height=800, width="100%")