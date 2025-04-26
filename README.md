# MammoCLIP

MammoCLIP is a deep learning framework for analyzing mammography images and clinical data, built on top of PyTorch. It provides tools for data extraction, preprocessing, model training, and evaluation in the context of BI-RADS assessments and breast cancer detection.

## Features

- Data extraction and preprocessing from DICOM/JPEG images
- Customizable data loaders and augmentation pipelines
- Config-driven training (`config.yaml`)
- Modular model architectures in `custom/model.py`
- Training and monitoring scripts (`train.py`, `monitor_train.py`)

## Repository Structure

```plaintext
.
├── README.md                # Project overview and instructions
├── LICENSE                  # MIT License
├── config.yaml              # Default configuration file
├── extract-birads-fast.py   # Fast BI-RADS extraction script
├── train.py                 # Training entry point
├── monitor_train.py         # Real-time training logger
├── custom/                  # Custom modules (data, model, preprocessing)
└── utils/                   # Utility functions (logging, metrics, seeds)
```  

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mahbodez/MammoCLIP.git
   cd MammoCLIP
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # (create this file with your libs)
   ```

3. Configure `config.yaml` with paths to your data and training parameters.

## Usage

- **Data Extraction**  
  ```bash
  python extract-birads-fast.py --config config.yaml
  ```

- **Training**  
  ```bash
  python train.py --config config.yaml
  ```

- **Monitoring**  
  ```bash
  python monitor_train.py --log-dir logs/
  ```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
