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
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
import threading
import questionary as Q

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


def extract_info(
    ref_df: pd.DataFrame,
    report_col: str,
    number_to_extract: int,
    template: str,
    keys: list[str],
    provider: str,
    model: str,
):
    get_response: callable = None
    thread_name = threading.current_thread().name
    if "openai" in provider.lower():
        client = OpenAI()
        get_response = partial(openai_get_response, client=client, model=model)
    elif "google" in provider.lower():
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        get_response = partial(google_get_response, client=client, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    tqdm.write(
        f"{Fore.BLUE}[{thread_name}]{Style.RESET_ALL} Using {provider} model: {model}"
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
    model: str,
    template: str,
    keys: list[str],
):
    batch_size = total_number_to_extract // number_of_threads

    whole_df = ref_df.copy(deep=True)
    sub_dfs = [
        whole_df.loc[idxs].copy()
        for idxs in np.array_split(whole_df.index, number_of_threads)
    ]

    results = [None] * number_of_threads
    threads = []

    def thread_worker(
        i, sub_df, report_col, batch_size, provider, model, template, keys
    ):
        df, n = extract_info(
            ref_df=sub_df,
            report_col=report_col,
            number_to_extract=batch_size,
            provider=provider,
            model=model,
            template=template,
            keys=keys,
        )
        if n == 0:
            tqdm.write(f"{Fore.YELLOW}[thread{i}] No more reports to extract")
        results[i] = df

    for i, sub_df in enumerate(sub_dfs):
        t = threading.Thread(
            target=thread_worker,
            args=(i, sub_df, report_col, batch_size, provider, model, template, keys),
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


def find_csv_files(root: str = "./data"):
    # find all csv files in the root directory
    csv_files = [
        os.path.join(root, f)
        for f in os.listdir(root)
        if os.path.isfile(os.path.join(root, f)) and f.lower().endswith(".csv")
    ]
    return csv_files


def main():
    console.print(
        Panel.fit(
            Text("\U0001f4c4 CSV File Selection", style="bold cyan"),
            subtitle="Use arrow keys",
            style="green",
        )
    )
    ref_csv = Q.select(
        "Select the reference CSV file",
        choices=find_csv_files(),
        use_indicator=True,
        qmark="❓",
    ).ask()
    dst_csv = Q.select(
        "Select the destination CSV file",
        choices=find_csv_files() + ["same"],
        default="same",
        use_indicator=True,
        qmark="❓",
    ).ask()
    if dst_csv == "same":
        dst_csv = ref_csv
    if not os.path.exists(ref_csv):
        raise FileNotFoundError(f"File not found: {ref_csv}")
    if not os.path.exists(dst_csv):
        raise FileNotFoundError(f"File not found: {dst_csv}")
    ref_df = pd.read_csv(ref_csv)
    console.print(
        Panel.fit(
            Text("\u2699\ufe0f YAML Configuration File Selection", style="bold cyan"),
            subtitle="Fill out",
            style="green",
        )
    )
    cfg_file = Prompt.ask(
        "Enter the path to the YAML configuration file",
        default="extract-cfg.yaml",
        console=console,
    )
    if not os.path.exists(cfg_file) or not cfg_file.lower().endswith(".yaml"):
        raise FileNotFoundError(f"File not found: {cfg_file}")
    with open(cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if "templates" not in cfg or "models" not in cfg:
        raise ValueError("Invalid configuration file. Missing 'templates' or 'models'.")
    templates = cfg["templates"]
    models = cfg["models"]

    console.print(
        Panel.fit(
            Text("\U0001f4dc Report Column Selection", style="bold cyan"),
            subtitle="Use arrow keys",
            style="green",
        )
    )

    report_col = Q.select(
        "Select the report column",
        choices=ref_df.columns.tolist(),
        default="report",
        use_indicator=True,
        qmark="❓",
    ).ask()

    console.print(
        Panel.fit(
            Text("\U0001f9f6 Extraction Parameters", style="bold cyan"),
            subtitle="Fill out",
            style="green",
        )
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

    console.print(
        Panel.fit(
            Text("\U0001f916 Template and Model Selection", style="bold cyan"),
            subtitle="Use arrow keys",
            style="green",
        )
    )

    template = Q.select(
        "Select the template to use for extraction",
        choices=list(templates.keys()),
        default=list(templates.keys())[0],
        use_indicator=True,
        qmark="❓",
    ).ask()
    if template not in templates:
        raise ValueError(f"Unsupported template: {template}")
    template = templates[template]
    template_prompt = template["prompt"]
    template_keys = template["keys"]

    model = Q.select(
        "Select the model to use for extraction",
        choices=list(models.keys()),
        default=list(models.keys())[0],
        use_indicator=True,
        qmark="❓",
    ).ask()
    if model not in models:
        raise ValueError(f"Unsupported model: {model}")
    provider = models[model]["provider"]

    confirm = Prompt.ask(
        "Do you want to proceed with the extraction? (yes/no)",
        default="yes",
        console=console,
    )
    if confirm.lower() != "yes":
        console.print("Extraction process aborted.")
        return

    multi_threaded_fill(
        ref_df=ref_df,
        output_file=dst_csv,
        report_col=report_col,
        total_number_to_extract=total_number_to_extract,
        number_of_threads=number_of_threads,
        provider=provider,
        model=model,
        template=template_prompt,
        keys=template_keys,
    )

    tqdm.write(f"{Fore.GREEN}Extraction process completed.")


if __name__ == "__main__":
    main()
