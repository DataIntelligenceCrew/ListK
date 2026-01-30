"""
    Implements some of the prompt formatting methods from LOTUS specifically for text
"""
class Prompt_Formatting:
    """
        Formats promtps for Filter and Top K for generic LLMs (not listwise rankers)
    """

    def label_text(
        document: str,
    )->dict[str,str]:
        """
            Tags a string as text
        """
        return {"text": document}
    
    def cot_formatter(
        reasoning: str,
        answer: str,
    )->str:
        """
            returns a string with the formated reasoning and answer for COT
        """
        return f"""Reasoning:\n{reasoning}\n\nAnswer: {answer}"""
    
    def non_cot_formatter(
        answer: str
    )-> str:
        """
            returns a string with the formatted answer when not using COT
        """
        return f"""Answer: {answer}"""
    
    def cot_prompt_formatter(
        reasoning: str = "",
        answer: str = ""
    )->str:
        """
            generates generic prompt addition (for system in filter) for COT
        """
        reasoning = f"<Your reasoning here. {reasoning}>"
        answer = f"<Your answer here. {answer}>"
        return f"""Let's think step by step. Use the following format to provide your answer:
            {Prompt_Formatting.cot_formatter(reasoning, answer)}
        """
    
    def non_cot_prompt_formatter(
        answer: str = ""
    )-> str:
        """
            generates generic prompt addition (for system in filter) when not using COT
        """
        answer = f"<Your answer here. {answer}>"
        return f"""Use the following format to provide your answer:
                {Prompt_Formatting.non_cot_formatter(answer)}
                """
    
    def user_message_formatter(
        document: str,
        user_instruction: str = ""
    )->dict[str,str]:
        """
            Formats a document and some user instruction into a user section of a prompt
        """
        return {
            "role": "user",
            "content": f"Context:\n{document}\n\n{user_instruction}",
        }
    
    def filter_formatter(
        document: str,
        query: str,
        cot: bool = False,
        reasoning_instruction: str = ""
    )->list[dict[str,str]]:
        """
            Takes a document (string) and query (string) and returns a formatted filter prompt.
        """
        answer_instructions = "The answer should be either True or False"
        sys_instruction = """The user will provide a claim and some relevant context.
        Your job is to determine whether the claim is true for the given context.
        """
        if cot:
            sys_instruction += Prompt_Formatting.cot_prompt_formatter(
                reasoning=reasoning_instruction, answer=answer_instructions
        )
        else:
            sys_instruction += Prompt_Formatting.non_cot_prompt_formatter(answer=answer_instructions)
        messages = [
            {"role": "system", "content": sys_instruction},
        ]
        messages.append(Prompt_Formatting.user_message_formatter(document, f"Claim: {query}"))
        return messages
    
    def pairwise_top_k_formatter(
        document1: str,
        document2: str,
        query: str,
        cot: bool = False
    )->list[dict[str, str]]:
        """
            takes two documents (strings) and a query (string) and returns a formatted top k prompt
            (for pairwise comparison).
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
        prompt = [{"type": "text", "text": f"Question: {query}\n"}]
        for idx, doc in enumerate([document1, document2]):
            labeled_text = Prompt_Formatting.label_text(doc)
            prompt += [{"type": "text", "text": f"\nDocument {idx+1}:\n{labeled_text}"}]
        messages: list[dict[str, str]] = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        return messages
