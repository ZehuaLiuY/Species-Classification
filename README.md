# Species and Long-Tail Recognition

## Model Weight

For the fine-tuned model weights of the results illustrated on the paper, please contact me via email at: liuzehuazxy@163.com

## Project Structure

The repository is structured as follows:

```
Species-Classification/
│
├─ envs/                    # Environment configuration files
│
├─ finetune/                # Fine-tuning Pre-trained model
│
├─ reference/               # Reference materials and literature
│
├─ src/                     # Proposed Method Source code
│
├─ tools/                   # Additional tools and scripts
│
│
├─ .gitignore               # Git ignore file
├─ README.md                # Project documentation
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ZehuaLiuY/Species-Classification.git
cd Species-Classification
```

### 2. Create and Activate a Virtual Environment

Use **Conda** or any other virtual environment tool.

```bash
# Example using Conda
conda create -n species_classification python=3.9 -y
conda activate species_classification
pip install pytorch-wildlife
```
## Data preparation 
Put the paths(Absolute is recommended) in the initialisation of the dataset class, normally in the main function of the training script.

## Training and Testing

### **Fine-tune the Model (Single GPU)**

For single-GPU training, run:

```bash
cd finetune
python FineTuneMega.py
```

This will train the model using the provided dataset and configurations.

### **Train the Model with DDP (Multi-GPU Acceleration)**

### **Train the Model on BlueCrystal4 (BC4)**

To run the training on **BlueCrystal4**, submit the SLURM job:

```bash
sbatch run.sh
```
It will use 11 GPUs to train the model.

### **Test the Model**

To evaluate the trained model, run:

```bash
python test.py --model_path="./models/focal_loss/best_model.pth"
```

This script loads the best-trained model and evaluates it on the test dataset.
