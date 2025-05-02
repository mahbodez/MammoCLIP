import pandas as pd
import numpy as np
from tqdm import tqdm
import colorama
from colorama import Fore, Style
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from google import genai
from google.genai import types
import time
from functools import partial
import os
import json
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


def extract_json_from_response(response: str):
    """
    Extract JSON from the response string.
    Args:
        response (str): The response string.
    Returns:
        dict: The extracted JSON object.
    Raises:
        ValueError: If the response does not contain valid JSON.
    """
    try:
        json_response = response[response.index("{") : response.rindex("}") + 1]
        json_response = json.loads(json_response)
    except (json.JSONDecodeError, ValueError):
        raise ValueError("Response does not contain a valid JSON.")
    return json_response


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def openai_get_response(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 64,
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
        )
        .choices[0]
        .message.content
    )
    return response


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def google_get_response(
    client: genai.Client,
    model: str,
    prompt: str,
    rpm: int = 20,
    max_tokens: int = 64,
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


template_birads = """
You are a clinical natural language processing (NLP) assistant specialized in radiology report understanding.

Extract the following fields from the mammogram report:

- "left_birads": BI-RADS score for the left breast (0–6 or -1), ignore characters like 'a' or 'b' or 'c'.
- "right_birads": BI-RADS score for the right breast (0–6 or -1), ignore characters like 'a' or 'b' or 'c'.

Return a JSON with these keys. Only extract what is explicitly stated. If uncertain or not mentioned, use -1. Do not infer.

Report:

<report>
{report}
</report>

Make sure to return a JSON response. Do not include any other text or explanation.

"""

template_composition = """
You are a clinical natural language processing (NLP) assistant specialized in radiology report understanding.

Extract the following fields from the mammogram report:

- "composition": The breast composition category ("A", "B", "C", "D").

Return a valid JSON with this key. 
Only extract what is explicitly stated. 
If uncertain or not mentioned, use "Z" as value. 
Do not infer.

Report:

<report>
{report}
</report>

Make sure to return a valid and parseable JSON response. Do not include any other text or explanation.
"""

template = template_composition
keys = ["composition"]

models = {
    "openai-mini": "gpt-4.1-mini",
    "openai": "gpt-4.1",
    "google": "models/gemini-2.0-flash",
}


def extract_info(
    ref_df: pd.DataFrame,
    report_col: str,
    number_to_extract: int,
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
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    tqdm.write(
        f"{Fore.BLUE}[{thread_name}]{Style.RESET_ALL} Using {provider} model: {models[provider]}"
    )

    df = ref_df.copy(deep=True)
    # if the extracted cols are non-existent, create them
    for key in keys:
        if key not in df.columns:
            df[key] = None
    # sample a subset of patients to extract
    conditions = (
        (df[report_col].notna())
        & (df[keys].isna()).all(axis=1)
        & (df[report_col].astype(str).str.strip().ne(""))
        & (df["l_cc"].notna())
        & (df["r_cc"].notna())
        & (df["l_mlo"].notna())
        & (df["r_mlo"].notna())
    )
    # ------ Check if there are any patients to extract from ------
    n = min(
        number_to_extract,
        df.loc[conditions].shape[0],
    )
    if n == 0:
        tqdm.write(f"{Fore.YELLOW}[{thread_name}] No patients to extract")
        return df, 0
    elif n < number_to_extract:
        tqdm.write(
            f"{Fore.YELLOW}[{thread_name}] Only {n} patients available for extraction"
        )
    # ------ Sample n patients to extract ------
    tqdm.write(f"{Fore.BLUE}[{thread_name}] Extracting {n} patients")
    indices = df.loc[conditions].sample(n=n, replace=False).index
    # ------ Extract the reports with tqdm (fixed position per thread) ------
    try:
        position = int(thread_name.replace("thread", ""))
    except ValueError:
        position = 0
    pbar = tqdm(
        indices,
        desc=f"[{thread_name}] Extracting from report",
        total=len(indices),
        position=position,
        leave=False,
    )
    for index in pbar:
        pbar.set_description(f"[{thread_name}] Extracting from report {index}")
        report = df.at[index, report_col]
        prompt = compile_prompt(template, report=report)
        try:
            response = get_response(prompt=prompt)
            response = dict(extract_json_from_response(response))
            # Try to fill in the extracted fields
            for key in keys:
                if key in response:
                    df.at[index, key] = response[key]
                else:
                    continue
        except Exception as e:
            tqdm.write(f"{Fore.RED}[{thread_name}] Error: {str(e)[:50]}")
            continue

    # ------ Return the augmented reports ------
    tqdm.write(f"{Fore.GREEN}[{thread_name}] Extraction complete!")
    return df, n


def multi_threaded_fill(
    ref_df: pd.DataFrame,
    output_file: str,
    report_col: str,
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

    def thread_worker(i, sub_df, report_col, batch_size, provider):
        df, n = extract_info(
            ref_df=sub_df,
            report_col=report_col,
            number_to_extract=batch_size,
            provider=provider,
        )
        if n == 0:
            tqdm.write(f"{Fore.YELLOW}[thread{i}] No more reports to extract")
        results[i] = df

    for i, sub_df in enumerate(sub_dfs):
        t = threading.Thread(
            target=thread_worker,
            args=(i, sub_df, report_col, batch_size, provider),
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
            for key in keys:
                if key in row and row[key] is not None:
                    whole_df.at[idx, key] = row[key]
        # write out
        whole_df.to_csv(output_file, index=False)
        tqdm.write(f"{Fore.GREEN}Saved combined results to {output_file}")
    else:
        tqdm.write(f"{Fore.YELLOW}No results to combine.")


def main():
    ref_csv = Prompt.ask(
        "Enter the path to the source CSV file with the reports",
        default="",
        console=console,
    )
    dst_csv = Prompt.ask(
        "Enter the path to the destination CSV file",
        default="same",
        console=console,
    )
    if dst_csv == "same":
        dst_csv = ref_csv
    if not os.path.exists(ref_csv):
        raise FileNotFoundError(f"File not found: {ref_csv}")
    if not os.path.exists(dst_csv):
        raise FileNotFoundError(f"File not found: {dst_csv}")
    if not Confirm(
        prompt="Do you want to overwrite the source file?",
        console=console,
    ):
        console.clear()
        main()
        return

    report_col = Prompt.ask(
        "Enter the name of the report column",
        default="report",
        console=console,
    )
    number_of_threads = Prompt.ask(
        "Enter the number of threads to use for extraction",
        default="10",
        console=console,
    )
    number_of_threads = int(number_of_threads)
    if number_of_threads < 1:
        raise ValueError("Number of threads must be at least 1")

    total_number_to_extract = Prompt.ask(
        "Enter the total number of reports to extract",
        default=1000,
        console=console,
    )
    total_number_to_extract = int(total_number_to_extract)
    if total_number_to_extract % number_of_threads != 0:
        raise ValueError(
            f"Total number of reports to extract ({total_number_to_extract}) must be divisible by the number of threads ({number_of_threads})"
        )

    provider = Prompt.ask(
        "Enter the provider to use for extraction (openai-mini, openai, google)",
        default=list(models.keys())[0],
        choices=list(models.keys()),
        console=console,
    )
    if provider not in models:
        raise ValueError(f"Unsupported provider: {provider}")

    multi_threaded_fill(
        ref_df=pd.read_csv(ref_csv),
        output_file=dst_csv,
        report_col=report_col,
        total_number_to_extract=total_number_to_extract,
        number_of_threads=number_of_threads,
        provider=provider,
    )

    tqdm.write(f"{Fore.GREEN}Extraction process completed.")


if __name__ == "__main__":
    main()
