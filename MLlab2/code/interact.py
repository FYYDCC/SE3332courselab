from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import torch

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', type=str, default="/root/autodl-tmp/distilGpt2",
                        help='the path to load fine-tuned model')
    parser.add_argument('--max_length', type=int, default=64,
                        help='maximum length for code generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature for sampling-based code geneeration')
    parser.add_argument(
        "--use_cuda", action="store_true", help="inference with gpu?"
    )

    args = parser.parse_args()

    # load fine-tunned model and tokenizer from path
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    model.eval()
    if args.use_cuda:
        model.to("cuda")

    # now the fine-tunned model supports two programming languages, namely, python and java
    def lang_select():
        lang = ""
        while lang not in ["python", "java"]:
            print('Enter the programming language you prefer (python or java)')
            lang = input(">>> ").lower()
        return lang


    lang = lang_select()

    context = ""
    while context != "exit":
        print(f'You are using {lang} now. Enter the context code (exit or change_lang)')
        context = input(">>> ")

        if context == "change_lang":
            lang = lang_select()

            print(f"You are using {lang} now. Enter the context code")
            context = input(">>> ")

        input_ids = tokenizer.encode("<python> " + context,
                                     return_tensors='pt') if lang == "python" else tokenizer.encode(
            "<java> " + context, return_tensors='pt')

         # 创建与输入形状相同的全1张量，作为注意力遮罩
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

        outputs = model.generate(input_ids=input_ids.to("cuda") if args.use_cuda else input_ids,
                                 max_length=args.max_length,
                                 do_sample=True,  # 设置为 True 以启用采样模式
                                 temperature=args.temperature,
                                 attention_mask=attention_mask,
                                 num_return_sequences=1)
        for i in range(1):
            decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
            # ends with occurence of double new lines (to meet the convention of code completion)
            if "\n\n" in decoded:
                decoded = decoded[:decoded.index("\n\n")]

            print('Generated {}: {}'.format(i, decoded))
