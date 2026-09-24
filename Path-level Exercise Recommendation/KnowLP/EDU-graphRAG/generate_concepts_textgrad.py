import json
import os
import argparse
import tqdm
import textgrad as tg

from textgrad.engine.openai import ChatOpenAI


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_PATH = os.path.dirname(SCRIPT_DIR)
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate concept descriptions for the KnowLP GraphRAG pipeline."
    )
    parser.add_argument("--dataset", default="assist17")
    parser.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    return parser.parse_args()


def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment before running this script.")

    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_API_BASE)

    vertex_file = os.path.join(
        args.base_path, "data", "dataProcess", args.dataset, "graph_vertex.json"
    )
    output_dir = os.path.join(args.base_path, "EDU-graphRAG", "ragtest", "input")
    output_file = os.path.join(output_dir, f"{args.dataset}_concepts.txt")

    os.makedirs(output_dir, exist_ok=True)
    
    # Configure TextGrad to use the OpenAI-compatible Qwen endpoint.
    engine = ChatOpenAI(model_string=args.model)
    tg.set_backward_engine(engine)
    
    # Load the concept names produced by the common preprocessing stage.
    with open(vertex_file, 'r', encoding='utf-8') as f:
        vertex_dict = json.load(f)
    
    concept_names = list(vertex_dict.keys())
    print(f"Loaded {len(concept_names)} concepts; starting TextGrad optimization...")

    # Resume safely by skipping concepts already present in the output file.
    existing_concepts = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            for c in concept_names:
                if f"[{c}]:" in content:
                    existing_concepts.add(c)
                    
    all_concepts_str = ", ".join(concept_names)
                    
    # Process concepts in batches using the original begin/end logic.
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i in tqdm.tqdm(range(0, len(concept_names), args.batch_size)):
            begin = i
            end = min(i + args.batch_size - 1, len(concept_names) - 1)
            batch_concepts = concept_names[begin:end+1]
            
            # Skip batches that have already been written.
            if all(c in existing_concepts for c in batch_concepts):
                continue
                
            try:
                # Preserve the prompt structure used by the original implementation.
                template_str = "\n".join([f"[{c}]: " for c in batch_concepts])
                prompt_text = (
                    "You are an expert in the field of education and are responsible for the detailed interpretation of knowledge concepts. "
                    f"You must analyze the following {len(batch_concepts)} specific concepts: {', '.join(batch_concepts)}.\n\n"
                    "To help you find relationships, here is the full list of all existing concepts in the database:\n"
                    f"<full_list>{all_concepts_str}</full_list>\n\n"
                    "CRITICAL OUTPUT FORMAT:\n"
                    f"You MUST output exactly {len(batch_concepts)} paragraphs. You MUST use the exact skeleton below and just append your analysis after the colon.\n"
                    "Do NOT add any Markdown headers. Do NOT analyze any concepts not in the skeleton.\n"
                    "Skeleton:\n"
                    f"{template_str}"
                )
                
                initial_prompt = tg.Variable(
                    prompt_text, 
                    requires_grad=False,
                    role_description="Initial prompt for knowledge concepts interpretation"
                )
                
                # Generate the initial draft and enable textual gradients.
                generator = tg.BlackboxLLM(engine)
                draft = generator(initial_prompt)
                draft.requires_grad = True 
                
                # Preserve the original evaluation instruction.
                evaluation_instruction = (
                    "Make the generated analysis smarter, more logical and accurate, more specific and discriminative rather than vague, "
                    "and avoid ambiguity. Ensure that the analysis of knowledge concepts is correct. "
                    f"CRITICAL FIX: The draft MUST contain exactly {len(batch_concepts)} items, specifically for these exact concepts: {', '.join(batch_concepts)}. "
                    "If any concept from this list is missing, you MUST add it back. If any unrequested concept is analyzed, remove it. "
                    "The format `[knowledge concept]: [analysis]` must be strictly followed."
                )
                loss_fn = tg.TextLoss(evaluation_instruction)
                
                # Define the optimizer.
                optimizer = tg.TGD(parameters=[draft])
                
                # Run textual gradient descent.
                for _ in range(args.max_iterations):
                    loss = loss_fn(draft)
                    loss.backward()
                    optimizer.step()
                    
                # Store the final optimized text.
                final_explanation = draft.value
                
                # Write the text consumed by GraphRAG.
                f_out.write(final_explanation + "\n\n")
                f_out.flush()
            
            except Exception as e:
                print(f"Failed to process range {begin}-{end}: {e}")
                # Fall back to one ordinary generation when TGD fails.
                try:
                    print(f"Trying a single fallback generation for range {begin}-{end}...")
                    fallback = generator(initial_prompt)
                    final_explanation = fallback.value
                    f_out.write(final_explanation + "\n\n")
                    f_out.flush()
                    print(f"Fallback output saved for range {begin}-{end}; continuing.")
                    continue
                except Exception as e2:
                    print(f"Fallback failed for range {begin}-{end}; skipping it: {e2}")
                    continue

    print(f"\nGeneration complete. Combined output saved to:\n{output_file}")

if __name__ == "__main__":
    main(parse_args())
