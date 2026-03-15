from typing import List, Tuple
import torch
import re


def format_tokens(tokens):
    """
    Format the tokens by adding specific values to the end of each input.
    Parameters:
        tokens: Tokens to be formatted, typically from a tokenizer.
    Returns:
        tokens: Formatted tokens with additional values appended.
    """

    # Formatting tokens
    values_to_add = torch.tensor([50256, 1303, 21017])

    # Expand the values to match the number of input examples
    values_expanded = values_to_add.unsqueeze(0).expand(tokens.input_ids.size(0), -1)

    # Concatenate the values as new columns
    tokens.input_ids = torch.cat((tokens.input_ids, values_expanded), dim=1)

    return tokens


def read_and_format_tokens(data_path, tokenizer, n=None):
    """
    Read and format tokens from a file.

    Parameters:
        data_path (str): Path to the .txt file containing input data.
        tokenizer: Tokenizer to use for encoding the input data.
        n (int): Number of examples to read from the file. If -1, read
            all examples.
    Returns:
        tokens: Formatted tokens ready for model input.
    """

    # Load input as each row of the file processed_valid.txt
    with open(data_path, "r") as f:
        # Read one row at a time
        texts = f.readlines()

    # Add initial space and remove new line character
    texts = [" " + text.strip() + " " for text in texts]

    if n is not None:
        # Take only the first n examples
        texts = texts[:n]

    # Tokenize and format
    tokens = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=11)
    tokens = format_tokens(tokens)

    return tokens


def read_operands(data_path, flip_operands=False, as_int=True):
    """
    Read operands from a .txt file.

    Parameters:
        data_path (str): Path to the .txt file containing input data.
        flip_operands (bool): Whether to flip the operands digit-wise.
            If True, the operands will be reversed.
            Should be True when reading from a file where operands are already in reverse order.
        as_int (bool): If True, convert operands to integers.
            Not appying flip operand might remove place values, so both settings being true is not recommended.
    Returns:
        operands (List[Tuple[str, str]]): List of tuples.
    """

    # Load input as each row of the file processed_valid.txt
    with open(data_path, "r") as f:
        # Read one row at a time
        texts = f.readlines()

    # Remove spaces
    texts = [text.replace(" ", "").replace("\n", "").split("*") for text in texts]

    # Flip operands if needed
    if flip_operands:
        texts = [(a[::-1], b[::-1]) for a, b in texts]

    if as_int:
        # Convert operands to integers
        texts = [(int(a), int(b)) for a, b in texts]

    return texts


def calculate_correct_ans(tokenizer, tokens):

    # Decode the tokens
    _input = tokenizer.decode(tokens, skip_special_tokens=True)

    # Parse the first digit of each number
    dig_1 = _input.split(" * ")[0].replace(" ", "")[0]
    dig_2 = _input.split(" * ")[1].replace(" ", "")[0]

    # Calculate the correct answer
    correct = int(dig_1) * int(dig_2)

    # Return last digit of the correct answer
    return " " + str(correct)[-1]


def multiply(a: int, b: int, return_reverse=False) -> str:
    """
    Multiply a, b, optionally return the resulting solution in inverse order.
    returns a str.
    """
    ans = str(a * b)
    if return_reverse:
        return ans[::-1]
    return ans


def get_ci_from_operands(a, b, i):
    """
    Get i'th digit from solution.
    Assumes a, b are in "correct" order.
    """
    ans = multiply(a, b, return_reverse=True)
    return int(ans[i])


def tokens_to_operands(tokens: torch.Tensor, tokenizer) -> List[Tuple[int, int]]:
    """
    Convert tokens to operands (a, b) pairs.
    The returned (a, b) pairs in correct decimal order.
    Parameters:
        tokens: Tensor of token IDs.
        tokenizer: Tokenizer used to decode the tokens.
    Returns:
        List of tuples containing operands (a, b).
    """
    # Decode the tokens
    decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)

    # Extract operands from the decoded strings
    operands = []
    for text in decoded:

        # Isolate operands
        a, b = text.split("####")[0].split(" * ")

        # Get correct decimal order and
        # Remove spaces and convert to integers
        a = int(a[::-1].strip().replace(" ", ""))
        b = int(b[::-1].strip().replace(" ", ""))

        # Append the operands as a tuple
        operands.append((a, b))

    return operands


def get_ci(a, b, i):
    """
    Get i'th digit from solution. Assumes that a and b are integers in the correct decimal format.
    Parameters:
        a (int): First operand.
        b (int): Second operand.
        i (int): Index of the digit to retrieve from the REVERSED product of a and b.
    """
    ans = multiply(a, b, return_reverse=True)
    if i < 0 or i >= len(ans):
        raise ValueError(f"Index {i} is out of bounds for the answer {ans}.")

    return ans[i]


def format_operands(
    operands: List[Tuple[int, int]],
    tokenizer,
    flip_operands=False,
    add_special_tokens=True,
):
    """
    Parameters:
        operands: [(a, b), ...]
        tokenizer
        flip_operands: if (a, b) need to be flipped digit-wise.
            In the case of reading from file, operands are already flipped so
            should be set to False.
        add_special_tokens: whether to add [50256, 1303, 21017] to the end of each input.
    """
    if flip_operands:
        formatted_operands = [
            " " + " ".join(str(a))[::-1] + " * " + " ".join(str(b))[::-1] + " "
            for a, b in operands
        ]
    else:
        formatted_operands = [
            " " + " ".join(str(a)) + " * " + " ".join(str(b)) + " " for a, b in operands
        ]

    tokens = tokenizer(
        formatted_operands,
        return_tensors="pt",
        padding="max_length",
        max_length=11,
    )

    if add_special_tokens:
        # Formatting tokens
        values_to_add = torch.tensor([50256, 1303, 21017])

        # Expand the values to match the number of input examples
        values_expanded = values_to_add.unsqueeze(0).expand(
            tokens.input_ids.size(0), -1
        )

        # Concatenate the values as new columns
        tokens.input_ids = torch.cat((tokens.input_ids, values_expanded), dim=1)
    return tokens


def prompt_ci_raw_format_batch(raw_data: List[str], ci: int, tokenizer):
    """
    raw_data: List[str] in the format of data stored in file:
        Each item (ex: 5 6 3 2 * 7 4 3 4) is **flipped** already.
    """
    operands_correct_order = [
        (
            int(x.split(" * ")[0].replace(" ", "")[::-1]),
            int(x.split(" * ")[1].replace(" ", "")[::-1]),
        )
        for x in raw_data
    ]

    # Construct original prompt
    formatted = [" " + sample.strip() + " " for sample in raw_data]
    tokens = tokenizer(
        formatted, return_tensors="pt", padding="max_length", max_length=11
    )
    values_to_add = torch.tensor([50256, 1303, 21017])
    values_expanded = values_to_add.unsqueeze(0).expand(tokens.input_ids.size(0), -1)
    tokens.input_ids = torch.cat((tokens.input_ids, values_expanded), dim=1)

    if ci == 0:
        return tokens.input_ids

    answers = [multiply(a, b, return_reverse=True) for a, b in operands_correct_order]

    max_len = len(str(9999 * 9999))  # 8
    for idx, _ans in enumerate(answers):
        if len(_ans) < max_len:
            answers[idx] = _ans + "0" * (max_len - len(_ans))

    i_digits = [" ".join(x[:ci]) for x in answers]
    if i_digits[0] != "":
        i_digits = [" " + x for x in i_digits]

    augment_tokens = tokenizer(i_digits, return_tensors="pt")
    # [batch, seq_len]
    prompt_token_ids = torch.cat(
        (tokens.input_ids, augment_tokens.input_ids), dim=1
    ).long()
    return prompt_token_ids


def get_ci(input, i):
    """
    Get i'th digit from solution.
    Assumes input is in the form of
    "a0 ... * b0 ..." and a, b are in "flipped" order.
    """
    dig_1 = input.split(" * ")[0].replace(" ", "")
    dig_2 = input.split(" * ")[1].replace(" ", "")
    dig_1 = int(dig_1[::-1])
    dig_2 = int(dig_2[::-1])
    ci = get_ci_from_operands(dig_1, dig_2, i)
    return ci


def extract_answer(text):
    split_pattern = "####"
    if split_pattern not in text:
        return text.strip().replace(",", "")
    else:
        _, ans = text.strip().split("####", 1)
        ans = "####" + ans
        ans = ans.strip().replace(",", "")
        return ans


def get_ith_a_or_b_digit(input, a_or_b: str, i: int):
    assert a_or_b in ["a", "b"], "a_or_b must be either 'a' or 'b'"
    if a_or_b == "a":
        dig = int(input.split(" * ")[0].replace(" ", "")[i])
    else:
        dig = int(input.split(" * ")[1].replace(" ", "")[i])
    return dig


def prompt_ci_operands(
    operands: List[Tuple[int, int]], i: int, tokenizer, device="cpu"
) -> Tuple[List[str], torch.LongTensor]:
    """
    Generate prompts for c_i.
    operands: [(a, b), ...]
    """
    answers = [multiply(a, b, return_reverse=True) for a, b in operands]
    suffixes = ["" for _ in answers]
    if i >= 1:
        suffixes = [" " + " ".join(ans[:i]) for ans in answers]

    prompt_txts = [
        " " + " ".join(str(a))[::-1] + " * " + " ".join(str(b))[::-1] + " "
        for a, b in operands
    ]
    eos = tokenizer.eos_token
    prompt_txts = [
        f"{txt}{eos}{eos} ####{suffix}" for txt, suffix in zip(prompt_txts, suffixes)
    ]

    # [batch, seq_len]
    prompt_token_ids = tokenizer(prompt_txts, return_tensors="pt", padding=True).input_ids
    prompt_token_ids = prompt_token_ids.to(device)
    return prompt_txts, prompt_token_ids


