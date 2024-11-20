# 🎵 Mono2Stereo AI

**Mono2Stereo AI** provides training and inference tools for converting mono music into stereo using an UNet based AI model.

### 🔧 Setup the Environment

#### 🌐 Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

- **Windows:**

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **macOS and Linux:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 📦 Install the Required Packages

Make sure you have python >= 3.9 and <= 3.10

```bash
pip install .
```

### 🪟 Additional Step for Windows Users

To ensure Gradio uses GPU/CUDA and not default to CPU, uninstall and reinstall `torch`, `torchvision`, and `torchaudio` with the correct CUDA version:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ⚙️ Configuration

A sample `config.json` is included in the root directory. Customize it to specify directories for custom models and outputs (.wav and .mid files will be stored here):

```json
{
    "sample_rate": 48000,
    "chunk_size": 240000,
    "num_epochs": 50,
    "batch_size": 4,
    "learning_rate": 0.001,
    "original_dataset_dir": "C:/Users/dick/Documents/MonoToStereoTrainingDS/original",
    "mono_dir": "C:/Users/dick/Documents/MonoToStereoTrainingDS/mono",
    "stereo_dir": "C:/Users/dick/Documents/MonoToStereoTrainingDS/stereo",
    "model_save_path": "model/unet_audio_model.pth",
    "device": "cuda"
}
```

## 🖥️ Usage

### 🎚️ Prepare the training dataset

