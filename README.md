# A Deep Learning Model Based on Mamba for Automatic Segmentation in Cervical Cancer Brachytherapy

## Quick Start

To get started, install the required dependencies:

```bash
pip install -r requirements.txt
```

For the most stable installation of Mamba-related libraries on Linux, follow these steps:

1. **Create a Conda Virtual Environment:**

   ```bash
   conda create -n mamba python=3.10
   ```

2. **Activate the Environment:**

   ```bash
   conda activate mamba
   ```

3. **Install PyTorch (version 2.3.1) along with torchvision and torchaudio:**

   ```bash
   conda install cudatoolkit=11.8 -c nvidia
   pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl
   ```

4. **Offline Installation of `causal-conv1d` and `mamba-ssm` Libraries:**

   - **`causal-conv1d` (v1.4.0):**
     - Download the appropriate wheel file from the [causal-conv1d releases page](https://github.com/Dao-AILab/causal-conv1d/releases) corresponding to your system's specifications (Python version, CUDA version, etc.).

   - **`mamba-ssm` (v2.2.2):**
     - Download the suitable wheel file from the [mamba releases page](https://github.com/state-spaces/mamba/releases) that matches your system's configuration.

5. **Install the Downloaded Wheel Files:**

   ```bash
   pip install causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ```

**Note:** Ensure that the wheel files you download match your system's Python version, CUDA version, and PyTorch version to avoid compatibility issues. 