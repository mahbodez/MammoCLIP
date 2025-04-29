import pandas as pd
import numpy as np
from tqdm import tqdm
import colorama
from colorama import Fore, Style
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from google import genai
from google.genai import types
from ollama import Client
import time
from functools import partial
import os
from rich.console import Console
from rich.prompt import Prompt, Confirm
import threading

console = Console()
colorama.init(autoreset=True)


def compile_prompt(
    template: str,
    **kwargs,
):
    """
    Compile a prompt template with the given keyword arguments.

    Args:
        template (str): The prompt template string.
        **kwargs: Keyword arguments to fill in the template.

    Returns:
        str: The compiled prompt.
    """
    return template.format(**kwargs)


def decode_response(
    respone: str,
    enclosing: tuple = ("<", ">"),
):
    """
    Decode a response string by removing enclosing characters.

    Args:
        respone (str): The response string to decode.
        enclosing (tuple): A tuple of two characters that enclose the response.

    Returns:
        str: The decoded response.
    """
    # if the response contains thinking tokens, remove them
    # they are usually enclosed between <think> and </think>
    if "<think>" in respone and "</think>" in respone:
        end = respone.rindex("</think>")
        respone = respone[end + len("</think>") :]
    start, end = enclosing
    try:
        result = respone.split(start)[1].split(end)[0]
    except Exception:
        result = respone
    return result.strip()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def openai_get_response(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.2,
):
    response = (
        client.chat.completions.create(
            model=model,
            messages=(
                [
                    {"role": "user", "content": prompt},
                ]
            ),
            timeout=15,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        .choices[0]
        .message.content
    )
    return response


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def ollama_get_response(
    client: Client,
    model: str,
    prompt: str,
):
    response = client.chat(
        model=model,
        messages=(
            [
                {"role": "user", "content": prompt},
            ]
        ),
    ).message.content
    return response


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def google_get_response(
    client: genai.Client,
    model: str,
    prompt: str,
    rpm: int = 20,
    max_tokens: int = 256,
):
    time.sleep(60 / rpm)
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            # http_options=types.HttpOptions(timeout=15),
        ),
    ).text
    return response


template = """
You are a highly experienced radiologist specializing in breast imaging.
Your task is to paraphrase the provided mammogram report text, strictly adhering to these instructions:
Preserve all medical facts and clinical findings from the original text.
Do NOT add or omit any medical details.
Maintain the original diagnostic meaning, clarity, and specificity.
Only alter the sentence structure, phrasing, or synonyms where medically equivalent and clearly appropriate.
Ensure the paraphrase is phrased naturally and professionally, as expected in clinical mammography reports.
Preserve key medical terminology (e.g., BIRADS categories, anatomical terms, pathology findings), but you may substitute medically approved synonyms if and only if they fully maintain the original meaning.
Try your best to change the words and sentence structure but not the meaning.

The mammogram report in <> is as follows:

<{report}>

Your paraphrased report should be enclosed in <>
"""

models = {
    "openai-mini": "gpt-4.1-mini",
    "openai-4.1": "gpt-4.1",
    "openai-4o": "gpt-4o",
    "google": "models/gemini-2.0-flash",
    "ollama:qwen": "qwen3",
}


def augment_reports(
    ref_df: pd.DataFrame,
    report_col: str,
    aug_report_col: str,
    number_to_augment: int,
    provider: str,
):
    get_response: callable = None
    thread_name = threading.current_thread().name
    if "openai" in provider:
        client = OpenAI()
        get_response = partial(
            openai_get_response, client=client, model=models[provider]
        )
    elif "google" in provider:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        get_response = partial(
            google_get_response, client=client, model=models[provider]
        )
    elif "ollama" in provider:
        client = Client(host="http://adir-pc.local:11435")
        get_response = partial(
            ollama_get_response, client=client, model=models[provider]
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    tqdm.write(
        f"{Fore.BLUE}[{thread_name}]{Style.RESET_ALL} Using {provider} model: {models[provider]}"
    )

    df = ref_df.copy(deep=True)
    # if the aug col is non-existent, create it
    if aug_report_col not in df.columns:
        df[aug_report_col] = None
    # sample a subset of patients to extract
    conditions = (
        (df[report_col].notna())
        & (df[aug_report_col].isna())
        & (df[report_col].str.strip().ne(""))
        & (df["l_cc"].notna())
        & (df["r_cc"].notna())
        & (df["l_mlo"].notna())
        & (df["r_mlo"].notna())
    )
    # ------ Check if there are any patients to extract from ------
    n = min(
        number_to_augment,
        df.loc[conditions].shape[0],
    )
    if n == 0:
        tqdm.write(f"{Fore.YELLOW}[{thread_name}] No patients to augment")
        return df, 0
    elif n < number_to_augment:
        tqdm.write(
            f"{Fore.YELLOW}[{thread_name}] Only {n} patients available for augmentation."
        )
    # ------ Sample n patients to extract ------
    tqdm.write(f"{Fore.BLUE}[{thread_name}] Augmenting {n} patients")
    indices = df.loc[conditions].sample(n=n, random_state=42, replace=False).index
    # ------ Extract the reports with tqdm (fixed position per thread) ------
    try:
        position = int(thread_name.replace("thread", ""))
    except ValueError:
        position = 0
    pbar = tqdm(
        indices,
        desc=f"[{thread_name}] Augmenting report",
        total=len(indices),
        position=position,
        leave=False,
    )
    for index in pbar:
        pbar.set_description(f"[{thread_name}] Augmenting report {index}")
        report = df.at[index, report_col]
        prompt = compile_prompt(template, report=report)
        try:
            response = get_response(prompt=prompt)
            response = decode_response(response)
            # Try to fill in the augmented report column
            df.at[index, aug_report_col] = response if response.strip() != "" else None
        except Exception as e:
            tqdm.write(f"{Fore.RED}[{thread_name}] Error: {str(e)[:50]}")
            continue

    # ------ Return the augmented reports ------
    tqdm.write(f"{Fore.GREEN}[{thread_name}] Augmentation complete!")
    return df, n


def multi_threaded_fill(
    ref_df: pd.DataFrame,
    output_file: str,
    report_col: str,
    aug_report_col: str,
    total_number_to_extract: int,
    number_of_threads: int,
    provider: str,
):
    batch_size = total_number_to_extract // number_of_threads

    whole_df = ref_df.copy(deep=True)
    sub_dfs = [
        whole_df.loc[idxs].copy()
        for idxs in np.array_split(whole_df.index, number_of_threads)
    ]

    results = [None] * number_of_threads
    threads = []

    def thread_worker(i, sub_df, report_col, aug_report_col, batch_size, provider):
        df, n = augment_reports(
            ref_df=sub_df,
            report_col=report_col,
            aug_report_col=aug_report_col,
            number_to_augment=batch_size,
            provider=provider,
        )
        if n == 0:
            tqdm.write(f"{Fore.YELLOW}[thread{i}] No reports to augment!")
        results[i] = df

    for i, sub_df in enumerate(sub_dfs):
        t = threading.Thread(
            target=thread_worker,
            args=(i, sub_df, report_col, aug_report_col, batch_size, provider),
            name=f"thread{i}",
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    valid_results = [df for df in results if df is not None]

    # ------ Combine the results into a single DataFrame ------
    if valid_results:
        combined_df = pd.concat(valid_results, ignore_index=False)
        # update
        for idx, row in combined_df.iterrows():
            whole_df.at[idx, aug_report_col] = row[aug_report_col]
        # write out
        whole_df.to_csv(output_file, index=False)
        tqdm.write(f"{Fore.GREEN}Saved combined results to {output_file}")
    else:
        tqdm.write(f"{Fore.YELLOW}No results to combine.")


def main():
    ref_csv = Prompt.ask(
        "Enter the path to the CSV file with the reports",
        default="./data/mammo-aug-oai-08-ggl-05.csv",
        console=console,
    )
    dest_csv = Prompt.ask(
        "Enter the path to save the augmented CSV file",
        default="./data/mammo-aug-oai-08-ggl-05.csv",
        console=console,
    )

    if not ref_csv.endswith(".csv"):
        raise ValueError("The reference CSV file must have a .csv extension")
    if not dest_csv.endswith(".csv"):
        raise ValueError("The destination CSV file must have a .csv extension")
    if os.path.exists(dest_csv):
        overwrite = Confirm.ask(
            "The destination file already exists. Do you want to overwrite it?",
            default=False,
            console=console,
        )
        if not overwrite:
            tqdm.write(f"{Fore.RED}Exiting without overwriting the file.")
            return

    if not os.path.exists(ref_csv):
        raise FileNotFoundError(f"The reference CSV file does not exist: {ref_csv}")

    report_col = Prompt.ask(
        "Enter the name of the report column",
        default="report",
        console=console,
    )
    aug_report_col = Prompt.ask(
        "Enter the name of the augmented report column",
        default="aug_report",
        console=console,
    )
    if report_col == aug_report_col:
        raise ValueError(
            "The report column and the augmented report column must be different"
        )
    number_of_threads = Prompt.ask(
        "Enter the number of threads to use for augmentation",
        default=1,
        console=console,
    )
    number_of_threads = int(number_of_threads)
    if number_of_threads < 1:
        raise ValueError("Number of threads must be at least 1")
    if number_of_threads > 60:
        raise ValueError("Number of threads must be less than 60")

    total_number_to_extract = Prompt.ask(
        "Enter the total number of reports to augment",
        default=1000,
        console=console,
    )
    total_number_to_extract = int(total_number_to_extract)
    if total_number_to_extract % number_of_threads != 0:
        raise ValueError(
            f"Total number of reports to augment ({total_number_to_extract}) must be divisible by the number of threads ({number_of_threads})"
        )

    provider = Prompt.ask(
        f"Enter the provider/model to use for augmentation",
        default=list(models.keys())[0],
        choices=list(models.keys()),
        console=console,
    )
    if provider not in models:
        raise ValueError(f"Unsupported provider: {provider}")

    multi_threaded_fill(
        ref_df=pd.read_csv(ref_csv),
        output_file=dest_csv,
        report_col=report_col,
        aug_report_col=aug_report_col,
        total_number_to_extract=total_number_to_extract,
        number_of_threads=number_of_threads,
        provider=provider,
    )

    tqdm.write(f"{Fore.GREEN}Extraction process completed.")


if __name__ == "__main__":
    main()
