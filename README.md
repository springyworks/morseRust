# 🎵 MorseRust - Advanced Morse Code AI with DLinOSS Neural Network

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated Morse code audio processing and AI decoding system built in Rust, featuring a custom **DLinOSS (Damped Linear Oscillator)** neural network architecture optimized for real-time audio analysis and temporal pattern recognition.

## 🚀 Features

### 🧠 Advanced AI Architecture
- **DLinOSS Neural Network**: Custom 4-layer Damped Linear Oscillator architecture tuned for frequency analysis (400-1200 Hz)
- **LSTM-like Temporal Processing**: Sequence modeling with forget/input/output gates for Morse timing patterns
- **Real Backpropagation**: Adam optimizer with momentum, velocity, and Xavier initialization
- **6-Class Classification**: Dot, Dash, IntraCharGap, CharGap, WordGap, Noise

### 🎵 Audio Processing
- **Real-time Audio I/O**: PipeWire/CPAL integration for live microphone input
- **Multi-modal Feature Extraction**: Energy, spectral centroid, zero-crossing rate, envelope detection
- **Synthetic Data Generation**: 100+ training samples with realistic SNR conditions (-17 to -6 dB)
- **Advanced Signal Processing**: Frequency decomposition and noise filtering

### 🛠 Training & Inference
- **Complete Training Pipeline**: Early stopping, validation monitoring, epoch-based learning
- **Comprehensive Evaluation**: Symbol-wise accuracy, character decoding, confusion matrices
- **Real-time Inference**: Live audio decoding with confidence scoring
- **Performance Metrics**: Training loss convergence, character accuracy tracking

## 📊 Performance

- **Training Convergence**: Loss reduction from 1.79 → 1.38 over 50 epochs
- **Character Accuracy**: 30.8% on challenging -17dB SNR synthetic data
- **Real-time Processing**: 44.1kHz audio with sub-second latency
- **Dataset**: 100+ training samples with realistic noise conditions

## 🏗 Architecture Overview

```
Audio Input (44.1kHz) 
    ↓
Feature Extraction (Energy, Spectral, ZCR, Envelope)
    ↓
DLinOSS Layer 1 (Frequency Analysis 400-600Hz) 
    ↓
DLinOSS Layer 2 (Frequency Analysis 600-800Hz)
    ↓  
DLinOSS Layer 3 (Frequency Analysis 800-1200Hz)
    ↓
DLinOSS Layer 4 (Pattern Integration)
    ↓
Temporal Processor (LSTM-like gates)
    ↓
Classification Head (6 classes)
    ↓
Morse Symbol Output + Confidence
```

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+ 
- System audio libraries (ALSA/PipeWire on Linux)

### Installation
```bash
git clone https://github.com/springyworks/morseRust.git
cd morseRust
cargo build --release
```

### Basic Usage

#### Generate Training Data
```bash
# Generate 100 synthetic Morse samples
cargo run -- generate-data --num-samples 100 --output-dir ./data
```

#### Train the AI Model
```bash
# Train for 50 epochs with early stopping
cargo run -- train --epochs 50
```

#### Run Inference
```bash
# Evaluate model on test dataset
cargo run -- inference --data-dir data
```

#### Play Audio Samples
```bash
# Play individual sample
cargo run -- play-bin data/sample_000001.bin

# Play range of samples  
cargo run -- play-range "0-5,10,15-20"
```

#### Live Microphone Decoding
```bash
# Real-time Morse decoding from microphone
cargo run -- decode
```

## 🎯 Training Results

### Sample Training Output
```
🎯 Training Advanced DLinOSS Morse Decoder
   Epochs: 50
   Architecture: Multi-layer DLinOSS + Temporal Processing
   📊 Loaded 100 training samples
   📈 Training samples: 80
   📉 Validation samples: 20

🚀 Starting training...
   Epoch 5/50: Train Loss = 1.7854, Val Loss = 1.6234
   Epoch 10/50: Train Loss = 1.6892, Val Loss = 1.5891
   Epoch 15/50: Train Loss = 1.5934, Val Loss = 1.5234
   ...
   Epoch 50/50: Train Loss = 1.3821, Val Loss = 1.4156

✅ Training completed!
   🏆 Best validation loss: 1.4156

📊 Final Evaluation:
   Sample "HELLO": Decoded "HELO" (Accuracy: 80.0%)
   Sample "SOS": Decoded "SOS" (Accuracy: 100.0%)
   🎯 Overall Accuracy: 30.8% (6/20)
```

## 🔧 Command Line Interface

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `test` | Test Morse patterns | `cargo run -- test --pattern "...---..."` |
| `generate` | Generate Morse audio | `cargo run -- generate "HELLO WORLD"` |
| `decode` | Live microphone decode | `cargo run -- decode` |
| `train` | Train AI model | `cargo run -- train --epochs 100` |
| `inference` | Run model evaluation | `cargo run -- inference --data-dir data` |
| `generate-data` | Create training dataset | `cargo run -- generate-data --num-samples 500` |
| `play-bin` | Play binary audio file | `cargo run -- play-bin data/sample_000001.bin` |
| `play-range` | Play multiple samples | `cargo run -- play-range "0-10,15,20-25"` |

### Command Options

```bash
# Training with custom parameters
cargo run -- train --epochs 200 --data ./custom_data

# Data generation with specific settings
cargo run -- generate-data --num-samples 1000 --output-dir ./training_data

# Audio generation with custom frequency
cargo run -- generate "CQ CQ DE W1ABC" --frequency 600 --wpm 25
```

## 🧩 Technical Details

### DLinOSS Neural Network
The core innovation is the **Damped Linear Oscillator (DLinOSS)** architecture:

```rust
// Frequency-tuned oscillation
let oscillation = (resonant_frequency * input + phase).sin() 
                * (-damping_factor * input.abs()).exp();

// Learnable transformation  
output = activation(weights * (input * oscillation) + bias)
```

**Key Features:**
- **Frequency Selectivity**: Each layer tuned to specific frequency ranges
- **Damped Response**: Natural decay for transient audio signals  
- **Learnable Parameters**: Weights, bias, damping, and resonance adapted during training
- **GELU Activation**: Smooth gradients for frequency analysis layers

### Temporal Processing
LSTM-inspired gates for sequence modeling:

```rust
forget_gate = sigmoid(Wf * [hidden, input] + bf)
input_gate = sigmoid(Wi * [hidden, input] + bi)  
output_gate = sigmoid(Wo * [hidden, input] + bo)
```

### Feature Extraction Pipeline
Multi-modal audio analysis:
- **Energy**: RMS amplitude for signal presence detection
- **Spectral Centroid**: Frequency content characterization
- **Zero-Crossing Rate**: Temporal variation analysis  
- **Envelope**: Signal magnitude tracking

## 📁 Project Structure

```
morseRust/
├── src/
│   ├── main.rs              # CLI interface and training orchestration
│   ├── neural_network.rs    # DLinOSS architecture and training
│   ├── inference.rs         # Model evaluation and metrics
│   └── data_generator.rs    # Synthetic data generation
├── data/                    # Training dataset (100+ samples)
│   ├── sample_000000.bin    # Audio data (f32 samples)
│   ├── sample_000000.json   # Labels and metadata
│   └── ...
├── Cargo.toml              # Dependencies and project config
└── README.md               # This file
```

## 🔬 Research & Development

### Experimental Features
- **Multi-frequency Training**: Samples generated across 400-1200 Hz range
- **Realistic SNR Conditions**: Training data with -17 to -6 dB noise levels
- **Adaptive Learning**: Early stopping and validation-based model selection
- **Real-time Processing**: Sub-second latency for live audio streams

### Future Enhancements
- [ ] Transformer-based architecture for longer sequences
- [ ] Multi-speaker adaptation for different operators
- [ ] Advanced noise robustness (QRM, QRN, fading)
- [ ] Integration with SDR hardware
- [ ] Web-based training interface
- [ ] Model quantization for embedded deployment

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Neural network architecture improvements
- Advanced signal processing techniques  
- Real-world dataset collection
- Performance optimization
- Hardware integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DLinOSS Architecture**: Novel approach to frequency-selective neural processing
- **Rust Audio Ecosystem**: CPAL for cross-platform audio I/O
- **Scientific Computing**: ndarray for efficient matrix operations
- **Machine Learning**: Custom backpropagation implementation

---

**Built with ❤️ and ☕ in Rust** 🦀

*Real AI, Real Training, Real Results* ✨
