# Nifty Midcap LSTM Predictions

This repository houses trained LSTM models used for predicting the trends in the Nifty Midcap index. The models are stored as `.pth` files, representing the weights of trained neural networks in PyTorch.

---

## Repository Contents

- **`trained_models/`**: This folder contains all the trained LSTM model files in `.pth` format.

---

## Getting Started

### **Prerequisites**

To use these models, you will need Python and PyTorch installed. You can install PyTorch by running:

```bash
pip install torch
```

### **Using the Models**

To load and use the models for predictions, follow these steps:

```python
import torch
import torch.nn as nn

# Assuming the architecture of your LSTM model is defined
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        # Define the layers of your model here
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load a model
model_path = 'trained_models/your_model_name.pth'
model = LSTMModel(input_size=..., hidden_size=..., num_layers=..., output_size=...)
model.load_state_dict(torch.load(model_path))
model.eval()

# Example: Predict using the model
# Make sure your input_data is formatted as required by your model
input_data = torch.randn(1, sequence_length, input_size)  # Example input
with torch.no_grad():
    predictions = model(input_data)
    print("Predicted values:", predictions)
```

---

## Contributing

Feel free to fork this repository and submit pull requests with improvements or contact me directly if you have specific suggestions or questions.

---

## License

This project is released under the MIT License - see the [LICENSE](LICENSE) file for details.

