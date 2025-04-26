from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict
import logging
import yaml
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class Config:
    project_dir: str = ""
    seed: int = 42
    csv_path: str = ""

    freeze_vision_model: bool = True
    freeze_text_model: bool = True

    training_params: dict = field(
        default_factory=lambda: dict(
            num_epochs=100,
            train_fraction=0.8,
            batch_size=8,
            lr_max=1e-4,
            lr_min=5e-7,
            lr_scheduler="wsd_schedule",
            warmup_fraction=0.05,
            steady_fraction=0.25,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            weight_decay=1e-6,
            mixed_precision="bf16",
        )
    )
    dl_workers: dict = field(
        default_factory=lambda: dict(
            train=4,
            val=4,
        )
    )

    train_ds: dict = field(
        default_factory=lambda: dict(
            class_="MammogramDataset",
            attrs_=dict(
                path_to_df="",
                weights_col="weight",
                pid_col="id",
                image_cols=["r_cc", "l_cc", "r_mlo", "l_mlo"],
                text_col="report",
                alt_text_cols=["aug_report"],
                image_preprocessor=dict(
                    class_="MammogramPreprocessor",
                    attrs_=dict(
                        output_size=(518, 518),
                        use_clahe=True,
                        extract_largest_cc=True,
                    ),
                ),
                transform_function=dict(
                    class_="MammogramTransform",
                    attrs_=dict(
                        size=(518, 518),
                        degrees=10,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        shear=(10, 10),
                        mean=(0.281,),
                        std=(0.217,),
                        noise_std=(0.0, 0.03),
                        is_validation=False,
                    ),
                ),
                tokenizer_kwargs=dict(
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ),
                alt_text_prob=0.66,
                cache_dir=".cache",
                tokenizer="microsoft/BiomedVLP-CXR-BERT-general",
            ),
        )
    )

    val_ds: dict = field(
        default_factory=lambda: dict(
            class_="MammogramDataset",
            attrs_=dict(
                path_to_df="",
                pid_col="id",
                image_cols=["r_cc", "l_cc", "r_mlo", "l_mlo"],
                text_col="report",
                # weights_col="weight",
                # alt_text_cols=["aug_report"],
                image_preprocessor=dict(
                    class_="MammogramPreprocessor",
                    attrs_=dict(
                        output_size=(518, 518),
                        use_clahe=True,
                        extract_largest_cc=True,
                    ),
                ),
                transform_function=dict(
                    class_="MammogramTransform",
                    attrs_=dict(
                        size=(518, 518),
                        degrees=10,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        shear=(20, 20),
                        mean=(0.281,),
                        std=(0.217,),
                        noise_std=(0.0, 0.03),
                        is_validation=True,
                    ),
                ),
                tokenizer_kwargs=dict(
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ),
                alt_text_prob=0.0,
                cache_dir=".cache",
                tokenizer="microsoft/BiomedVLP-CXR-BERT-general",
            ),
        )
    )

    eval_interval: int = 1  # epoch
    save_interval: int = 1  # epoch
    max_checkpoints: int = 5  # max number of checkpoints to keep

    pretrained_model_cfg: dict = field(
        default_factory=lambda: dict(
            vision_model_name_or_path="microsoft/rad-dino",
            text_model_name_or_path="microsoft/BiomedVLP-CXR-BERT-general",
            num_views=4,
            fusion_type="linear",
            verbose=False,
        )
    )

    def to_dict(self) -> Dict[str, Any]:
        def convert_value(value):
            if is_dataclass(value):
                return value.to_dict()  # Recursive call for nested dataclasses
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(v) for v in value]
            else:
                return value

        return {
            field: convert_value(getattr(self, field))
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """
        Create a Config instance from a dictionary, handling nested structures.
        """
        instance = cls()

        def convert_value(field_name, value):
            field_info = instance.__dataclass_fields__.get(field_name)
            field_type = field_info.type if field_info else None

            # 1) Nested dataclass instance already on `instance`?
            if isinstance(value, dict):
                current_attr = getattr(instance, field_name, None)
                if is_dataclass(current_attr):
                    return current_attr.__class__.from_dict(value)

                # 2) Or the field annotation itself is a dataclass?
                if field_type and is_dataclass(field_type):
                    return field_type.from_dict(value)

                # 3) Plain dict → recurse into its items
                return {k: convert_value(k, v) for k, v in value.items()}

            # List or tuple → convert each element
            if isinstance(value, (list, tuple)):
                return [convert_value(field_name, v) for v in value]

            # Primitive → return as is
            return value

        # Create a new instance of the dataclass
        # Iterate over the fields and set their values
        for field_name, field_value in d.items():
            if hasattr(instance, field_name):
                # Convert the value and set it on the instance
                converted_value = convert_value(field_name, field_value)
                setattr(instance, field_name, converted_value)
            else:
                logger.warning(
                    f"Field '{field_name}' not found in Config dataclass. Skipping."
                )
        return instance

    def __str__(self) -> str:
        def format_dict(d, indent=0):
            res = ""
            for k, v in d.items():
                if isinstance(v, dict):
                    res += " " * indent + f"{k}:\n" + format_dict(v, indent + 2)
                elif isinstance(v, (list, tuple)):
                    res += " " * indent + f"{k}:\n"
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            res += (
                                " " * (indent + 2)
                                + f"-\n"
                                + format_dict(item, indent + 4)
                            )
                        else:
                            res += " " * (indent + 2) + f"- {item}\n"
                else:
                    res += " " * indent + f"{k}: {v}\n"
            return res

        return "Config:\n" + format_dict(self.to_dict(), 2)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load a Config instance from a YAML file.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        if not yaml_path.endswith(".yaml") and not yaml_path.endswith(".yml"):
            raise ValueError("File must be a .yaml or .yml file")
        if not os.path.isfile(yaml_path):
            raise ValueError(f"Path is not a file: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return cls.from_dict(dict(data))

    def to_yaml(self, yaml_path: str) -> None:
        """
        Save the Config instance to a YAML file.
        """
        if not yaml_path.endswith(".yaml") and not yaml_path.endswith(".yml"):
            raise ValueError("File must be a .yaml or .yml file")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f)
        logger.info(f"Config saved to {yaml_path}")

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """
        Load a Config instance from a JSON file.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not json_path.endswith(".json"):
            raise ValueError("File must be a .json file")
        if not os.path.isfile(json_path):
            raise ValueError(f"Path is not a file: {json_path}")
        with open(json_path, "r") as f:
            data = dict(json.load(f))
        return cls.from_dict(data)

    def to_json(self, json_path: str) -> None:
        """
        Save the Config instance to a JSON file.
        """
        if not json_path.endswith(".json"):
            raise ValueError("File must be a .json file")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Config saved to {json_path}")
