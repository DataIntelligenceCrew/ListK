import re
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM

'''
    Here are some important LOTUS methods that turn a users natural language query into
    an LLM prompt. Note: LOTUS and Rerankers (which is a wrapper for Rank LLM) both require
    python to be version 3.10 or higher. As RankLLM uses python 3.11 that is what is assumed
    for now.
'''

'''
    Pulled from LOTUS task_instructions.py it outputs the following:

    Reasoning: reasoning

    Answer: answer
'''
def cot_formatter(reasoning, answer):
    return f"""Reasoning:\n{reasoning}\n\nAnswer: {answer}"""

'''
    Pulled from LOTUS task_instructions.py it outputs the following:

    Answer: answer
'''
def answer_only_formatter(answer):
    return f"""Answer: {answer}"""

'''
    Pulled from LOTUS task_instructions.py it outputs the following:

    Let's think step by step. Use the following format to provide your answer:
    Reasoning: <Your reasoning here. user reasoning instructions>

    Answer: <Your answer here. user answer instructions>
'''
def cot_prompt_formatter(reasoning_instructions: str = "", answer_instructions: str = "") -> str:
    reasoning_instructions = f"<Your reasoning here. {reasoning_instructions}>"
    answer_instructions = f"<Your answer here. {answer_instructions}>"
    return f"""Let's think step by step. Use the following format to provide your answer:
        {cot_formatter(reasoning_instructions, answer_instructions)}
        """

'''
    Pulled from LOTUS task_instructions.py it outputs the following:

    Use the following format to provide your answer:
    Answer: <Your answer here. user answer instructions>
'''
def non_cot_prompt_formatter(answer_instructions: str = "") -> str:
    answer_instructions = f"<Your answer here. {answer_instructions}>"
    return f"""Use the following format to provide your answer:
            {answer_only_formatter(answer_instructions)}
            """

'''
    For now I will assume only string inputs into this method pulled from task_instruction.py
    for simplicity of examples, but in practice the lotus version also supports a dictionary format for
    images. To get the string and image tuple you use the context formatter method.
'''
def context_formatter(multimodal_data: dict[str, Any] | str,
) -> tuple[str, list[dict[str, str]]]:
    '''
        If we get some string
    '''
    if isinstance(multimodal_data, str):
        '''
            We will simply return a tuple of the string and an empty list or
            text, []
        '''
        text = multimodal_data
        image_inputs: list[dict[str, str]] = []
    elif isinstance(multimodal_data, dict):
        '''
            We handle anything tagged as an image as so:
                We put the stuff tagged as image into a dictionary of the
                text and url associated with the image.
        '''
        image_data: dict[str, str] = multimodal_data.get("image", {})
        '''
            We then produce a tuple of two dictionaries one with the images text
            (key) and another with the images url for each image (for this method it
            would actually be only one image).
        '''
        _image_inputs: list[tuple[dict, dict]] = [
            (
                {
                    "type": "text",
                    "text": f"[{key.capitalize()}]: \n",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            )
            for key, base64_image in image_data.items()
        ]
        image_inputs = [m for image_input in _image_inputs for m in image_input]
        '''
            For text we simply pass anything tagged as text or nothing if there is no text
        '''
        text = multimodal_data["text"] or ""
    else:
        raise ValueError("multimodal_data must be a dictionary or a string")
    return text, image_inputs

def user_message_formatter(
    multimodal_data: dict[str, Any] | str,
    user_instruction_with_tag: str | None = None,
) -> dict[str, Any]:
    text, image_inputs = context_formatter(multimodal_data)
    #For the sake of example only this branch will ever occur (as we have no images)
    if not image_inputs or len(image_inputs) == 0:
        '''
            As such we output the following dictionary:
            {
                "role": "user",
                "content": "Context:
                text data

                user instruction with tag"
            }
        '''
        return {
            "role": "user",
            "content": f"Context:\n{text}\n\n{user_instruction_with_tag}",
        }
    '''
        If we do have images we first form a context of the following dicts:
        [
        {
            "type": "text",
            "text": "Content:
            text data"
        },
        (The tuple of image data with the key and the image url)
        ]
    '''
    content = [{"type": "text", "text": f"Context:\n{text}"}] + image_inputs
    #We then append the user instructions as well
    if user_instruction_with_tag:
        content.append({"type": "text", "text": f"\n\n{user_instruction_with_tag}"})
    #And return it in the following dictionary format
    return {
        "role": "user",
        "content": content,
    }

'''
    Pulled from LOTUS task_instructions.py and modified (this is filter, but it acts as one of two examples of how the natural language
    is processed). Also something to note is that LOTUS has a specific COT text for deepseek, but I am only using the normal COT
    text in these examples. 
'''
def filter_formatter(
    multimodal_data: dict[str, Any],
    user_instruction: str,
    cot: bool = False,
    reasoning_instructions: str = "",
) -> list[dict[str, str]]:
    answer_instructions = "The answer should be either True or False"
    sys_instruction = """The user will provide a claim and some relevant context.
    Your job is to determine whether the claim is true for the given context.
     """
    if cot:
        sys_instruction += cot_prompt_formatter(
            reasoning_instructions=reasoning_instructions, answer_instructions=answer_instructions
        )
    else:
        sys_instruction += non_cot_prompt_formatter(answer_instructions=answer_instructions)
    '''
        We feed our LLM a messages which are dictionaries of structure:

        {
            "role": "system",
            "content": sys_instruction
        }

        where the sys_instruction will be the sys_instruction string which in this case is:
            sys_instruction = """The user will provide a claim and some relevant context.
            Your job is to determine whether the claim is true for the given context.
            """
        With the cot_prompt_formatter or non_cot_prompt_formatter appended to the end which have the formats:

            Let's think step by step. Use the following format to provide your answer:
            Reasoning: <Your reasoning here. user reasoning instructions>

            Answer: <Your answer here. user answer instructions>
        
            and

            Use the following format to provide your answer:
            Answer: <Your answer here. user answer instructions>
    '''
    messages = [
        {"role": "system", "content": sys_instruction},
    ]
    '''
        LOTUS also allows for reasoning examples as well as answer examples, but for now lets assume we
        have the default with no examples. In this case our messages is only two items,
        the initial system message which includes system instructions, reasoning instruction (if using COT)
        and answer formatting instructions and an item containing the actual contents to be considered
        where the user instruction is considered within the second message. User_instruction
        in this case is considered the actual natural language prompt given by the user to filter items.
    '''
    messages.append(user_message_formatter(multimodal_data, f"Claim: {user_instruction}"))
    return messages


'''
    Pulled from LOTUS sem_topk.py and modified. Also something to note is that LOTUS has a specific COT text for deepseek, but I am only using the normal COT
    text in these examples. 
'''
def get_match_prompt_binary(
    doc1: dict[str, Any],
    doc2: dict[str, Any],
    user_instruction: str,
    cot: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate a binary comparison prompt for two documents.

    This function creates a prompt that asks the language model to compare two
    documents and select the one that better matches the user's instruction.
    It supports different reasoning strategies including chain-of-thought.

    Args:
        doc1 (dict[str, Any]): The first document to compare. Should contain
            multimodal information (text, images, etc.).
        doc2 (dict[str, Any]): The second document to compare. Should contain
            multimodal information (text, images, etc.).
        user_instruction (str): The natural language instruction that defines
            the comparison criteria.
        model (lotus.models.LM): The language model instance to use for comparison.
        strategy (ReasoningStrategy | None, optional): The reasoning strategy to use.
            Can be None, COT, or ZS_COT. Defaults to None.

    Returns:
        list[dict[str, Any]]: A list of message dictionaries formatted for the
            language model API.

    Example:
        >>> doc1 = {"text": "Machine learning tutorial"}
        >>> doc2 = {"text": "Data science guide"}
        >>> model = LM(model="gpt-4o")
        >>> prompt = get_match_prompt_binary(doc1, doc2, "Which is more relevant to AI?", model)
    """
    if cot:
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'First give your reasoning. Then you MUST end your output with "Answer: Document 1 or Document 2"\n'
            'You must pick a number and cannot say things like "None" or "Neither"\n'
            'Remember to explicitly state "Answer:" at the end before your choice.'
        )
    else:
        sys_prompt = (
            "Your job is to to select and return the most relevant document to the user's question.\n"
            "Carefully read the user's question and the two documents provided below.\n"
            'Respond only with the label of the document such as "Document NUMBER".\n'
            "NUMBER must be either 1 or 2, depending on which document is most relevant.\n"
            'You must pick a number and cannot say things like "None" or "Neither"'
        )
    '''
        Unlike filter we put the users natural language query into a seperate dictionary that is fed into the message.
    '''
    prompt = [{"type": "text", "text": f"Question: {user_instruction}\n"}]
    for idx, doc in enumerate([doc1, doc2]):
        content_text, content_image_inputs = context_formatter(doc)
        '''
            Likewise each document is its own dictionary
        '''
        prompt += [{"type": "text", "text": f"\nDocument {idx+1}:\n{content_text}"}, *content_image_inputs]

    '''
        For topk we utilize the system prompt to state the formatting of the prompt we will receive. The system
        is simply the instructions with the system prompt and the user prompt will be three dictionaries with the
        first being the natural language query, the second being the first document and the third the second document.
    '''
    messages: list[dict[str, Any]] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    return messages



'''
    Lets generate an example filter. Lets have the natural language query be:
        "The line of text contains the word LOTUS"
    and the input data to be:
        "LOTUS also allows for reasoning examples as well as answer examples, but for now lets assume we"
'''
example_query = "The line of text contains the word LOTUS"
example_data = {"text": "LOTUS also allows for reasoning examples as well as answer examples, but for now lets assume we"}
cotf = filter_formatter(example_data, example_query, True, "Example Reasoning Instructions")
noncotf = filter_formatter(example_data, example_query, False, "")
print("output of filter prompt generator with COT:")
print(cotf)
print("\n\n")
print("output of filter prompt generator without COT:")
print(noncotf)
print("\n\n")

'''
    Lets generate an example binary comparison with the provided lotus example.
'''
doc1 = {"text": "Machine learning tutorial"}
doc2 = {"text": "Data science guide"}
COTprompt = get_match_prompt_binary(doc1, doc2, "Which is more relevant to AI?", True)
nonCOTprompt = get_match_prompt_binary(doc1, doc2, "Which is more relevant to AI?", False)
print("output of binary comparison prompt generator with COT:")
print(COTprompt)
print("\n\n")
print("output of binary comparison prompt generator without COT:")
print(nonCOTprompt)
print("\n\n")

'''
    Lets pass this into an LLM to produce output. For examples I am going to run Qwen3 0.6B. Here is the two lines to
    load the model.
'''
tokenizer = AutoTokenizer.from_pretrained("/home/jshin/LLMtest/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("/home/jshin/LLMtest/Qwen3-0.6B")

'''
    We pass the messages we made into the model sequentially
'''
print("output of filter prompt with COT:")
messages = [cotf]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

print("output of filter prompt without COT:")
messages = [noncotf]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

print("output of binary comparison prompt without COT:")
messages = [COTprompt]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

print("output of binary comparison prompt without COT:")
messages = [nonCOTprompt]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
