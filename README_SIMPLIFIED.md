# Simplified Point Cloud to Wireframe Training

This project has been simplified and restructured for better understanding and maintainability.

## 🚀 Quick Start

### 1. Run Training
```bash
python train_multiple.py
```

### 2. Run Full Pipeline (Training + Evaluation)
```bash
python main.py
```

## 📁 File Structure

- **`train_multiple.py`** - Simplified training script with clear functions
- **`main.py`** - Complete pipeline including data loading, training, and evaluation
- **`demo_dataset/`** - Contains your point cloud (.xyz) and wireframe (.obj) files

## 🔧 Key Functions

### Training Functions
- **`train_model()`** - Main training function with clear parameters
- **`find_matching_files()`** - Finds matching .xyz and .obj files
- **`split_data_into_train_test()`** - Splits data into training/testing sets
- **`create_data_collator()`** - Handles batching of variable-sized data

### Evaluation Functions
- **`evaluate_model()`** - Evaluates trained model on test data

## 📊 Configuration

You can easily modify training parameters in `train_multiple.py`:

```python
# Training parameters
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
MAX_VERTICES = 32
BATCH_SIZE = 4
```

## 🎯 What the Code Does

1. **Data Loading**: Automatically finds and pairs your .xyz and .obj files
2. **Data Preprocessing**: Normalizes point clouds and wireframes
3. **Model Training**: Trains a neural network to predict wireframes from point clouds
4. **Evaluation**: Tests the model on unseen data
5. **Model Saving**: Saves the best trained model

## 🔍 Understanding the Code

### Before (Complex):
- Functions with unclear names like `make_pcwf_collate`
- Mixed Turkish/English comments
- Complex nested logic
- Hard to follow data flow

### After (Simple):
- Clear function names like `create_data_collator`
- English documentation throughout
- Step-by-step logic
- Easy to follow data flow

## 🚨 Troubleshooting

If you get import errors:
1. Make sure all required packages are installed: `pip install torch numpy`
2. Check that your dataset files are in the correct directories
3. Verify file paths in the configuration section

## 📈 Training Progress

The training script will show:
- Current epoch and loss
- Learning rate changes
- Best model performance
- Time elapsed

## 💾 Output

- **`trained_model.pth`** - Your trained model
- **Console logs** - Training progress and metrics
- **Evaluation results** - Model performance on test data

## 🎉 Success!

When training completes successfully, you'll see:
```
Training completed! Model saved as 'trained_model.pth'
```

Your model is now ready to convert new point clouds into wireframes!
