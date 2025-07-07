import re
from typing import List, Tuple


def find_mcq_end(string: str) -> Tuple[bool, int]:
    """Find the end of a multiple choice question.

    Args:
        string (str): The string to search for the end of a multiple choice question.

    Returns:
        Tuple[bool, int]: A tuple containing a boolean indicating whether a multiple choice question was found and the end position of the question.
    """
    patterns = [
        r"\([A-D]\)",
        r"[A-D][\.\)]",
        r"[a-d][\.\)]",
    ]

    combined_pattern = "|".join(patterns)
    matches = list(re.finditer(combined_pattern, string, re.IGNORECASE))

    if len(matches) >= 4:
        last_match = matches[-1]
        return True, last_match.end()

    return False, -1


def find_next_newline(string):
    single_newline = string.find("\n")
    double_newline = string.find("\n\n")

    if single_newline == -1 and double_newline == -1:
        return -1
    elif single_newline == -1:
        return double_newline
    elif double_newline == -1:
        return single_newline
    else:
        return min(single_newline, double_newline)


def generate_variants(prefixes: List[str]) -> List[str]:
    """Generate variants of the prefixes by adding colons, asterisks, and hash symbols.

    Args:
        prefixes (List[str]): A list of prefixes to generate variants for.

    Returns:
        List[str]: A list of variants of the prefixes.
    """
    variants = []
    for word in prefixes:
        clean_word = word.lstrip("#* ").rstrip(":")
        variants.append(f"{clean_word}:")
        variants.append(f"**{clean_word}**")
        variants.append(f"## {clean_word}:")

    return variants


def remove_prefix(string):
    # Remove numbers and alphabets with periods or brackets at the beginning
    string = re.sub(r"^(\d+\.|\w+\))\s*", "", string).strip()

    # Remove prefixes like "Task:", "Prompt:", "Question:"
    prefixes = [
        "Task",
        "Prompt",
        "Original Prompt",
        "Question",
        "The Problem",
        "Problem",
        "Scenario",
        "The senario",
        "Situation",
        "The situation",
        "Context",
        "Challenge",
        "Query",
        "Request",
        "Instructions",
        "Instruction",
        "Descriptions",
        "Description",
    ]
    prefixes = generate_variants(prefixes)

    prefix_patterns = [re.escape(prefix) for prefix in prefixes]

    prefix_patterns.append(r"Question \d+:")  # Question 1:, Question 2:, etc.
    prefix_patterns.append(r"Question \d+\.")  # Question 1., Question 2., etc.
    prefix_patterns.append(r"Part \d+:")  # Part 1:, Part 2:, etc.
    prefix_patterns.append(r"Q\d+\.")  # Q1., Q2., etc.
    prefix_patterns.append(r"Q\d+:")  # Q1:, Q2:, etc.

    combined_pattern = "|".join(prefix_patterns)

    match = re.match(f"^({combined_pattern})\s*", string, re.IGNORECASE)
    if match:
        return string[match.end() :].strip()

    return string


def instruction_post_process(instruction, model_path):
    if "gemma-2" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = [
            "Answer:",
            "Answers:",
            "The answer is",
            "Correct answer",
            "The correct answer",
            "Answer is",
            "Explanation:",
            "Here are some",
            "Solution Approach:",
            "Solution:",
        ]
        assistant_pattern = r"(?:" + "|".join(assistant_markers) + ")s?:"
        assistant_match = re.search(
            assistant_pattern, instruction
        )  # Exact match, no re.IGNORECASE

        # TODO
        # if is_mcq:
        #     print(f"MCQ detected: {instruction}")
        #     rest_of_string = instruction[end_pos:]
        #     newline_pos = find_next_newline(rest_of_string)
        #     if newline_pos != -1:
        #         instruction = instruction[:end_pos + newline_pos].strip()
        #     else:
        #         instruction = instruction.strip()
        #     print(f"Sanitized MCQ instruction: {instruction}")
        #     class_num = 0

        if instruction.startswith("*"):
            if "?" in instruction:
                instruction = instruction.split("?")[0].replace("*", "").strip() + "?"
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)

        if instruction.startswith('"'):
            if "?" in instruction:
                instruction = instruction.split("?")[0].replace('"', "").strip() + "?"
                class_num = 2.1
            else:
                instruction = instruction.split("\n")[0].replace('"', "").strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 2.2
        elif instruction.startswith("<b>"):
            instruction.split("\n")[0].replace("</b>", "").replace("<b>", "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[: assistant_match.start()].strip()
            instruction = (
                instruction.replace("**", "").strip()
                if instruction.find("**") == 1
                else instruction.strip()
            )
            class_num = 4
        elif instruction.split("\n")[0].strip().endswith(":"):
            colon_pos = instruction.split("\n")[0].strip().rfind(":")
            if "#" in instruction:
                instruction = instruction.split("#")[0].strip()
                instruction = (
                    instruction.replace("**", "").strip()
                    if instruction.find("**") == 1
                    else instruction.strip()
                )
                class_num = 5.1
            elif "?" in instruction:
                instruction = instruction.split("?")[0].strip() + "?"
                instruction = (
                    instruction.replace("**", "").strip()
                    if instruction.find("**") == 1
                    else instruction.strip()
                )
                class_num = 5.2
            else:
                instruction = instruction.split("\n")[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 5.3
        else:
            if "?" in instruction:
                instruction = instruction.split("?")[0].strip() + "?"
                instruction = (
                    instruction.replace("**", "").strip()
                    if instruction.find("**") == 1
                    else instruction.strip()
                )
                class_num = 6.1
            else:
                instruction = instruction.split("\n")[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num

    elif "llama-3" in model_path.lower():
        # remove the prefix
        instruction = remove_prefix(instruction)
        # find mcq problems
        is_mcq, end_pos = find_mcq_end(instruction)
        assistant_markers = [
            "Answer:",
            "Answers:",
            "The answer is",
            "Correct answer",
            "The correct answer",
            "Answer is",
            "Explanation:",
            "Here are some",
            "Solution Approach:",
            "Solution:",
        ]
        assistant_pattern = r"(?:" + "|".join(assistant_markers) + ")s?:"
        assistant_match = re.search(
            assistant_pattern, instruction
        )  # Exact match, no re.IGNORECASE
        # print(f"Assistant match: {assistant_match}")

        step_makers = ["# Step 1", "## Step 1", "### Step 1"]
        step_pattern = r"(?:" + "|".join(step_makers) + r"):?"
        step_match = re.search(
            step_pattern, instruction
        )  # Exact match, no re.IGNORECASE
        # print(f"Step match: {step_match}")

        # TODO
        # if is_mcq:
        #     print(f"MCQ detected: {instruction}")
        #     rest_of_string = instruction[end_pos:]
        #     newline_pos = find_next_newline(rest_of_string)
        #     if newline_pos != -1:
        #         instruction = instruction[:end_pos + newline_pos].strip()
        #     else:
        #         instruction = instruction.strip()
        #     print(f"Sanitized MCQ instruction: {instruction}")
        #     class_num = 0

        if instruction.startswith("*"):
            if "?" in instruction:
                instruction = instruction.split("?")[0].replace("*", "").strip() + "?"
                instruction = remove_prefix(instruction)
                class_num = 1
                return instruction, class_num

        instruction = remove_prefix(instruction)
        if instruction.startswith('"'):
            if "?" in instruction:
                instruction = instruction.split("?")[0].replace('"', "").strip() + "?"
                class_num = 2.1
            else:
                instruction = instruction.split("\n")[0].replace('"', "").strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 2.2
        elif instruction.startswith("<b>"):
            instruction.split("\n")[0].replace("</b>", "").replace("<b>", "").strip()
            instruction = instruction.replace("*", "").strip()
            class_num = 3
        elif assistant_match:
            instruction = instruction[: assistant_match.start()].strip()
            instruction = (
                instruction.replace("**", "").strip()
                if instruction.find("**") == 1
                else instruction.strip()
            )
            class_num = 4
        elif step_match:
            instruction = instruction[: step_match.start()].strip()
            instruction = (
                instruction.replace("**", "").strip()
                if instruction.find("**") == 1
                else instruction.strip()
            )
            class_num = 5
        elif instruction.split("\n")[0].strip().endswith(":"):
            colon_pos = instruction.split("\n")[0].strip().rfind(":")
            if "#" in instruction:
                instruction = instruction.split("#")[0].strip()
                instruction = (
                    instruction.replace("**", "").strip()
                    if instruction.find("**") == 1
                    else instruction.strip()
                )
                class_num = 6.1
            elif "?" in instruction:
                instruction = instruction.split("?")[0].strip() + "?"
                instruction = (
                    instruction.replace("**", "").strip()
                    if instruction.find("**") == 1
                    else instruction.strip()
                )
                class_num = 6.2
            else:
                instruction = instruction.split("\n")[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 6.3
        else:
            if "?" in instruction:
                instruction = instruction.split("?")[0].strip() + "?"
                instruction = (
                    instruction.replace("**", "").strip()
                    if instruction.find("**") == 1
                    else instruction.strip()
                )
                class_num = 99.1
            else:
                instruction = instruction.split("\n")[0].strip()
                instruction = instruction.replace("*", "").strip()
                class_num = 99.2

        # Remove prefixes again
        instruction = remove_prefix(instruction)

        return instruction, class_num

    else:
        return instruction, 0


# Define logits processor for llama-3.1 for de-markdown
def de_md_logits_processor_for_llama3_1(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999  # "#": 2,
        logits[567] = -9999.999  # "##": 567,
        logits[14711] = -9999.999  # "###": 14711,
        logits[827] = -9999.999  # "####": 827,

    return logits


# Define logits processor for flaming initial tokens
def flaming_tokens(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        # Slightly increase the temperature for the first token
        logits = logits / 1.2

    return logits


def remove_markdown_tags(
    content,
    remove_text_styles=True,  # 文本样式组（标题、强调、水平线）
    remove_code=True,  # 代码组（代码块、行内代码）
    remove_media=True,  # 媒体组（图片、表格、链接、脚注）
    remove_structures=True,  # 结构组（列表、引用）
    remove_html_tags=True,  # HTML 标签
):
    # 文本样式
    if remove_text_styles:
        content = re.sub(r"(^|\n)#+.*", "", content)  # 删除标题
        content = re.sub(r"\*\*([^\*]+)\*\*|\*([^\*]+)\*", r"\1\2", content)  # 删除强调
        content = re.sub(r"(^|\n)(---|\*\*\*|___)\n", "", content)  # 删除水平线

    # 代码
    if remove_code:
        content = re.sub(r"```[\s\S]*?```", "", content)  # 删除代码块
        content = re.sub(r"`[^`]+`", "", content)  # 删除行内代码

    # 媒体
    if remove_media:
        content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", content)  # 删除图片
        content = re.sub(r"(\|.*\|\n)+", "", content)  # 删除表格
        content = re.sub(
            r"\[([^\]]+)\]\([^\)]+\)", r"\1", content
        )  # 删除链接但保留文本
        content = re.sub(r"\[\^.+?\]:\s+.*", "", content)  # 删除脚注

    # 结构
    if remove_structures:
        content = re.sub(r"(^|\n)>.*", "", content)  # 删除引用
        content = re.sub(r"(^|\n)(-|\*|\+|\d+\.)\s+.*", "", content)  # 删除列表

    # HTML 标签
    if remove_html_tags:
        content = re.sub(r"<[^>]+>", "", content)

    # 删除多余的换行
    content = re.sub(r"\n{2,}", "\n", content)
    # 删除超长空格
    content = re.sub(r" {2,}", " ", content)

    return content.strip()
