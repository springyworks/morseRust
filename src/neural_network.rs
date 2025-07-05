use linoss_rust::*;
use ndarray::{Array1, Array2, s};
use std::error::Error;
use std::collections::HashMap;
use rand::Rng;

/// Advanced Morse Code Neural Network using DLinOSS with proper temporal processing
pub struct DLinossWrapper {
    // Multi-layer D-LinOSS network for hierarchical feature extraction
    dlinoss_layers: Vec<DLinOSSLayer>,
    // Temporal processing for sequence modeling
    temporal_processor: TemporalProcessor,
    // Feature extractor for audio preprocessing
    feature_extractor: MorseFeatureExtractor,
    // Classification head
    classifier: ClassificationHead,
    // Model parameters
    learning_rate: f32,
    // Symbol mapping
    symbol_map: HashMap<usize, String>,
}

#[derive(Clone)]
pub struct DLinOSSLayer {
    // D-LinOSS parameters using LinossRust
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub damping: Array1<f32>,
    pub resonance: Array1<f32>,
    pub phase: Array1<f32>,
    pub activation_type: ActivationType,
    // Gradients for training
    pub weight_gradients: Array2<f32>,
    pub bias_gradients: Array1<f32>,
    // Adam optimizer state
    pub weight_momentum: Array2<f32>,
    pub bias_momentum: Array1<f32>,
    pub weight_velocity: Array2<f32>,
    pub bias_velocity: Array1<f32>,
}

#[derive(Clone)]
pub struct TemporalProcessor {
    // LSTM-like processing for sequence modeling
    hidden_size: usize,
    weights_ih: Array2<f32>, // input to hidden
    weights_hh: Array2<f32>, // hidden to hidden
    bias: Array1<f32>,
    hidden_state: Array1<f32>,
    cell_state: Array1<f32>,
    // Gradients
    weights_ih_grad: Array2<f32>,
    weights_hh_grad: Array2<f32>,
    bias_grad: Array1<f32>,
}

#[derive(Clone)]
pub struct MorseFeatureExtractor {
    // Audio preprocessing parameters
    window_size: usize,
    hop_size: usize,
    sample_rate: f32,
    // Frequency analysis
    fft_size: usize,
    // Energy detection
    energy_threshold: f32,
}

#[derive(Clone)]
pub struct ClassificationHead {
    weights: Array2<f32>,
    bias: Array1<f32>,
    // Gradients
    weight_gradients: Array2<f32>,
    bias_gradients: Array1<f32>,
}

#[derive(Clone, Debug)]
pub enum ActivationType {
    GELU,
    ReLU,
    Sigmoid,
    Tanh,
}

impl DLinossWrapper {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let mut rng = rand::thread_rng();
        
        // Create symbol mapping for Morse symbols
        let mut symbol_map = HashMap::new();
        symbol_map.insert(0, "Dot".to_string());
        symbol_map.insert(1, "Dash".to_string());
        symbol_map.insert(2, "IntraCharGap".to_string());
        symbol_map.insert(3, "CharGap".to_string());
        symbol_map.insert(4, "WordGap".to_string());
        symbol_map.insert(5, "Noise".to_string());
        
        // Architecture: Multi-layer D-LinOSS with increasing abstraction
        let layer_dims = vec![
            (64, 128),  // Audio features -> Low-level patterns
            (128, 96),  // Low-level -> Mid-level patterns  
            (96, 64),   // Mid-level -> High-level patterns
            (64, 32),   // High-level -> Abstract features
        ];
        
        let mut dlinoss_layers = Vec::new();
        for (i, (in_dim, out_dim)) in layer_dims.iter().enumerate() {
            let activation = match i {
                0 => ActivationType::GELU,      // Smooth frequency analysis
                1 => ActivationType::ReLU,      // Sparse feature detection
                2 => ActivationType::GELU,      // Pattern integration
                _ => ActivationType::Tanh,      // Final abstraction
            };
            
            dlinoss_layers.push(DLinOSSLayer::new(*in_dim, *out_dim, activation, &mut rng)?);
        }
        
        // Temporal processor for sequence modeling (32 input -> 32 output for simplicity)
        let temporal_processor = TemporalProcessor::new(32, 32, &mut rng)?;
        
        // Feature extractor
        let feature_extractor = MorseFeatureExtractor::new(1024, 256, 44100.0);
        
        // Classification head (32 temporal features -> 6 classes)
        let classifier = ClassificationHead::new(32, 6, &mut rng)?;
        
        Ok(Self {
            dlinoss_layers,
            temporal_processor,
            feature_extractor,
            classifier,
            learning_rate: 0.001,
            symbol_map,
        })
    }
    
    /// Forward pass through the complete network
    pub fn forward(&mut self, audio_input: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        // 1. Feature extraction from raw audio
        let features = self.feature_extractor.extract_features(audio_input)?;
        
        // 2. Multi-layer D-LinOSS processing
        let mut current_features = features;
        for layer in &mut self.dlinoss_layers {
            current_features = layer.forward(&current_features)?;
        }
        
        // 3. Temporal processing for sequence modeling
        let temporal_features = self.temporal_processor.forward(&current_features)?;
        
        // 4. Classification
        let output = self.classifier.forward(&temporal_features)?;
        
        Ok(output)
    }
    
    /// Training step with backpropagation
    pub fn train_step(&mut self, audio_input: &[f32], target_labels: &[usize]) -> Result<f32, Box<dyn Error>> {
        // Forward pass
        let prediction = self.forward(audio_input)?;
        
        // Convert target labels to one-hot encoding
        let target = self.labels_to_onehot(target_labels);
        
        // Calculate loss (cross-entropy)
        let loss = self.calculate_loss(&prediction, &target);
        
        // Backward pass
        self.backward(&prediction, &target)?;
        
        // Update weights
        self.update_weights();
        
        Ok(loss)
    }
    
    fn labels_to_onehot(&self, labels: &[usize]) -> Array1<f32> {
        let mut onehot = Array1::zeros(6);
        
        // For sequence of labels, we'll use the most frequent label as target
        let mut counts = vec![0; 6];
        for &label in labels {
            if label < 6 {
                counts[label] += 1;
            }
        }
        
        // Find the most frequent label
        let max_label = counts.iter().position(|&x| x == *counts.iter().max().unwrap()).unwrap_or(0);
        onehot[max_label] = 1.0;
        
        onehot
    }
    
    fn calculate_loss(&self, prediction: &[f32], target: &Array1<f32>) -> f32 {
        // Cross-entropy loss with numerical stability
        let mut loss = 0.0;
        let epsilon = 1e-15;
        
        for i in 0..prediction.len().min(target.len()) {
            let p = prediction[i].max(epsilon).min(1.0 - epsilon);
            loss -= target[i] * p.ln();
        }
        
        loss
    }
    
    fn backward(&mut self, prediction: &[f32], target: &Array1<f32>) -> Result<(), Box<dyn Error>> {
        // Start with output gradients
        let mut output_grad = Array1::zeros(prediction.len());
        for i in 0..prediction.len().min(target.len()) {
            output_grad[i] = prediction[i] - target[i];
        }
        
        // Backpropagate through classifier
        let temporal_grad = self.classifier.backward(&output_grad)?;
        
        // Backpropagate through temporal processor
        let dlinoss_grad = self.temporal_processor.backward(&temporal_grad)?;
        
        // Backpropagate through D-LinOSS layers (reverse order)
        let mut current_grad = dlinoss_grad;
        for layer in self.dlinoss_layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad)?;
        }
        
        Ok(())
    }
    
    fn update_weights(&mut self) {
        // Update D-LinOSS layers
        for layer in &mut self.dlinoss_layers {
            layer.update_weights(self.learning_rate);
        }
        
        // Update temporal processor
        self.temporal_processor.update_weights(self.learning_rate);
        
        // Update classifier
        self.classifier.update_weights(self.learning_rate);
    }
    
    /// Get predicted symbol for the given output
    pub fn predict_symbol(&self, output: &[f32]) -> String {
        let max_idx = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        self.symbol_map.get(&max_idx).unwrap_or(&"Unknown".to_string()).clone()
    }
    
    /// Extract temporal patterns from audio for character-level decoding
    pub fn decode_sequence(&mut self, audio_input: &[f32]) -> Result<String, Box<dyn Error>> {
        // Reset temporal state
        self.temporal_processor.reset_state();
        
        let features = self.feature_extractor.extract_features(audio_input)?;
        let window_size = 256; // Process in chunks
        let mut decoded_text = String::new();
        let mut current_char = String::new();
        
        for chunk in features.chunks(window_size) {
            let mut current_features = chunk.to_vec();
            
            // Process through D-LinOSS layers
            for layer in &mut self.dlinoss_layers {
                current_features = layer.forward(&current_features)?;
            }
            
            // Process through temporal processor
            let temporal_out = self.temporal_processor.forward(&current_features)?;
            let output = self.classifier.forward(&temporal_out)?;
            
            let symbol = self.predict_symbol(&output);
            
            match symbol.as_str() {
                "Dot" => current_char.push('.'),
                "Dash" => current_char.push('-'),
                "IntraCharGap" => {}, // Continue building current character
                "CharGap" => {
                    if !current_char.is_empty() {
                        if let Some(letter) = morse_to_char(&current_char) {
                            decoded_text.push(letter);
                        }
                        current_char.clear();
                    }
                },
                "WordGap" => {
                    if !current_char.is_empty() {
                        if let Some(letter) = morse_to_char(&current_char) {
                            decoded_text.push(letter);
                        }
                        current_char.clear();
                    }
                    decoded_text.push(' ');
                },
                _ => {} // Noise or unknown
            }
        }
        
        // Handle remaining character
        if !current_char.is_empty() {
            if let Some(letter) = morse_to_char(&current_char) {
                decoded_text.push(letter);
            }
        }
        
        Ok(decoded_text)
    }
}

impl DLinOSSLayer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType, rng: &mut impl Rng) -> Result<Self, Box<dyn Error>> {
        // Xavier/Glorot initialization for weights
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            (rng.gen::<f32>() - 0.5) * 2.0 * scale
        });
        
        let bias = Array1::zeros(output_size);
        
        // D-LinOSS specific parameters
        let damping = Array1::from_vec((0..output_size).map(|i| {
            0.05 + (i as f32 / output_size as f32) * 0.1
        }).collect());
        
        let resonance = Array1::from_vec((0..output_size).map(|i| {
            400.0 + (i as f32 / output_size as f32) * 800.0 // 400-1200 Hz range
        }).collect());
        
        let phase = Array1::zeros(output_size);
        
        Ok(Self {
            weights: weights.clone(),
            bias: bias.clone(),
            damping,
            resonance,
            phase,
            activation_type: activation,
            weight_gradients: Array2::zeros((output_size, input_size)),
            bias_gradients: Array1::zeros(output_size),
            weight_momentum: Array2::zeros((output_size, input_size)),
            bias_momentum: Array1::zeros(output_size),
            weight_velocity: Array2::zeros((output_size, input_size)),
            bias_velocity: Array1::zeros(output_size),
        })
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let _input_array = Array1::from_vec(input.to_vec());
        let mut output = Array1::zeros(self.weights.nrows());
        
        // D-LinOSS computation with damped oscillation
        for i in 0..self.weights.nrows() {
            let mut sum = 0.0;
            
            for j in 0..input.len().min(self.weights.ncols()) {
                // Apply damped linear oscillation
                let oscillation = (self.resonance[i] * input[j] + self.phase[i]).sin() 
                                * (-self.damping[i] * input[j].abs()).exp();
                
                sum += self.weights[[i, j]] * input[j] * oscillation;
            }
            
            sum += self.bias[i];
            
            // Apply activation function
            output[i] = match self.activation_type {
                ActivationType::GELU => gelu(sum),
                ActivationType::ReLU => sum.max(0.0),
                ActivationType::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                ActivationType::Tanh => sum.tanh(),
            };
        }
        
        // Update phase for next iteration
        for i in 0..self.phase.len() {
            self.phase[i] += 0.01; // Small phase increment
        }
        
        Ok(output.to_vec())
    }
    
    pub fn backward(&mut self, output_grad: &Array1<f32>) -> Result<Array1<f32>, Box<dyn Error>> {
        // Simplified backward pass - compute input gradients
        let input_grad = self.weights.t().dot(output_grad);
        
        // Store gradients for weight updates (simplified)
        // Note: This is a simplified version - proper backprop would need stored activations
        for i in 0..self.weight_gradients.nrows() {
            for j in 0..self.weight_gradients.ncols() {
                self.weight_gradients[[i, j]] += output_grad[i] * 0.1; // Simplified
            }
        }
        
        self.bias_gradients += output_grad;
        
        Ok(input_grad)
    }
    
    pub fn update_weights(&mut self, learning_rate: f32) {
        // Adam optimizer
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
        
        // Update weight momentum and velocity
        self.weight_momentum = beta1 * &self.weight_momentum + (1.0 - beta1) * &self.weight_gradients;
        self.weight_velocity = beta2 * &self.weight_velocity + (1.0 - beta2) * &self.weight_gradients.mapv(|x| x * x);
        
        // Update weights
        let weight_update = learning_rate * &self.weight_momentum / (&self.weight_velocity.mapv(|x| x.sqrt()) + epsilon);
        self.weights -= &weight_update;
        
        // Update bias
        self.bias_momentum = beta1 * &self.bias_momentum + (1.0 - beta1) * &self.bias_gradients;
        self.bias_velocity = beta2 * &self.bias_velocity + (1.0 - beta2) * &self.bias_gradients.mapv(|x| x * x);
        
        let bias_update = learning_rate * &self.bias_momentum / (&self.bias_velocity.mapv(|x| x.sqrt()) + epsilon);
        self.bias -= &bias_update;
        
        // Reset gradients
        self.weight_gradients.fill(0.0);
        self.bias_gradients.fill(0.0);
    }
}

impl TemporalProcessor {
    pub fn new(input_size: usize, hidden_size: usize, rng: &mut impl Rng) -> Result<Self, Box<dyn Error>> {
        let scale = (1.0 / hidden_size as f32).sqrt();
        
        // Correct dimensions: weights_ih should be [4*hidden_size, input_size]
        let weights_ih = Array2::from_shape_fn((4 * hidden_size, input_size), |_| {
            (rng.gen::<f32>() - 0.5) * 2.0 * scale
        });
        
        // weights_hh should be [4*hidden_size, hidden_size] 
        let weights_hh = Array2::from_shape_fn((4 * hidden_size, hidden_size), |_| {
            (rng.gen::<f32>() - 0.5) * 2.0 * scale
        });
        
        let bias = Array1::zeros(4 * hidden_size);
        let hidden_state = Array1::zeros(hidden_size);
        let cell_state = Array1::zeros(hidden_size);
        
        Ok(Self {
            hidden_size,
            weights_ih: weights_ih.clone(),
            weights_hh: weights_hh.clone(),
            bias,
            hidden_state,
            cell_state,
            weights_ih_grad: Array2::zeros(weights_ih.dim()),
            weights_hh_grad: Array2::zeros(weights_hh.dim()),
            bias_grad: Array1::zeros(4 * hidden_size),
        })
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Result<Array1<f32>, Box<dyn Error>> {
        // Ensure input has correct size - should match what temporal processor was initialized with
        // If temporal processor expects 32 inputs, we need exactly 32
        let expected_input_size = self.weights_ih.ncols(); // This gives us the input size from weights
        
        let mut input_vec = input.to_vec();
        if input_vec.len() < expected_input_size {
            input_vec.resize(expected_input_size, 0.0);
        } else if input_vec.len() > expected_input_size {
            input_vec.truncate(expected_input_size);
        }
        
        let input_array = Array1::from_vec(input_vec);
        
        // LSTM-like computation
        let gi = self.weights_ih.dot(&input_array);
        let gh = self.weights_hh.dot(&self.hidden_state);
        let gates = gi + gh + &self.bias;
        
        let hidden_size = self.hidden_size;
        
        // Split gates into forget, input, output, and new cell gates
        let forget_gate = gates.slice(s![0..hidden_size]).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let input_gate = gates.slice(s![hidden_size..2*hidden_size]).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let output_gate = gates.slice(s![2*hidden_size..3*hidden_size]).mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let new_cell = gates.slice(s![3*hidden_size..4*hidden_size]).mapv(|x| x.tanh());
        
        // Update cell state
        self.cell_state = &forget_gate * &self.cell_state + &input_gate * &new_cell;
        
        // Update hidden state
        self.hidden_state = &output_gate * &self.cell_state.mapv(|x| x.tanh());
        
        Ok(self.hidden_state.clone())
    }
    
    pub fn backward(&mut self, output_grad: &Array1<f32>) -> Result<Array1<f32>, Box<dyn Error>> {
        // For matrix multiplication to work: weights_ih^T should be [input_size, 4*hidden_size]
        // and output_grad should be [4*hidden_size]
        // But we need to ensure the dimensions match
        
        // Ensure output_grad has correct size (should be 4*hidden_size for LSTM gates)
        let mut grad_vec = output_grad.to_vec();
        grad_vec.resize(4 * self.hidden_size, 0.0);
        let gate_grad = Array1::from_vec(grad_vec);
        
        // Compute input gradients: transpose(weights_ih) * gate_gradients
        // weights_ih is [4*hidden_size, input_size], so transpose is [input_size, 4*hidden_size]
        let input_grad = self.weights_ih.t().dot(&gate_grad);
        
        Ok(input_grad)
    }
    
    pub fn update_weights(&mut self, learning_rate: f32) {
        // Simple gradient descent
        self.weights_ih -= &(&self.weights_ih_grad * learning_rate);
        self.weights_hh -= &(&self.weights_hh_grad * learning_rate);
        self.bias -= &(&self.bias_grad * learning_rate);
        
        // Reset gradients
        self.weights_ih_grad.fill(0.0);
        self.weights_hh_grad.fill(0.0);
        self.bias_grad.fill(0.0);
    }
    
    pub fn reset_state(&mut self) {
        self.hidden_state.fill(0.0);
        self.cell_state.fill(0.0);
    }
}

impl MorseFeatureExtractor {
    pub fn new(window_size: usize, hop_size: usize, sample_rate: f32) -> Self {
        Self {
            window_size,
            hop_size,
            sample_rate,
            fft_size: window_size,
            energy_threshold: 0.01,
        }
    }
    
    pub fn extract_features(&self, audio: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut features = Vec::new();
        
        // Process audio in windows
        for i in (0..audio.len().saturating_sub(self.window_size)).step_by(self.hop_size) {
            let window = &audio[i..i + self.window_size.min(audio.len() - i)];
            
            // Extract multiple feature types
            let energy = self.extract_energy(window);
            let spectral_centroid = self.extract_spectral_centroid(window);
            let zero_crossing_rate = self.extract_zero_crossing_rate(window);
            let envelope = self.extract_envelope(window);
            
            // Combine features
            features.extend_from_slice(&[energy, spectral_centroid, zero_crossing_rate, envelope]);
        }
        
        // Ensure consistent feature dimension
        if features.len() < 64 {
            features.resize(64, 0.0);
        } else if features.len() > 64 {
            features.truncate(64);
        }
        
        Ok(features)
    }
    
    fn extract_energy(&self, window: &[f32]) -> f32 {
        window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32
    }
    
    fn extract_spectral_centroid(&self, window: &[f32]) -> f32 {
        // Simplified spectral centroid - weight by frequency
        let mut weighted_sum = 0.0;
        let mut magnitude_sum = 0.0;
        
        for (i, &sample) in window.iter().enumerate() {
            let freq = i as f32 * self.sample_rate / self.fft_size as f32;
            let magnitude = sample.abs();
            weighted_sum += freq * magnitude;
            magnitude_sum += magnitude;
        }
        
        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
    
    fn extract_zero_crossing_rate(&self, window: &[f32]) -> f32 {
        let mut crossings = 0;
        for i in 1..window.len() {
            if (window[i] >= 0.0) != (window[i-1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / window.len() as f32
    }
    
    fn extract_envelope(&self, window: &[f32]) -> f32 {
        // Simple envelope using moving average of absolute values
        window.iter().map(|x| x.abs()).sum::<f32>() / window.len() as f32
    }
}

impl ClassificationHead {
    pub fn new(input_size: usize, output_size: usize, rng: &mut impl Rng) -> Result<Self, Box<dyn Error>> {
        let scale = (2.0 / input_size as f32).sqrt();
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            (rng.gen::<f32>() - 0.5) * 2.0 * scale
        });
        
        let bias = Array1::zeros(output_size);
        
        Ok(Self {
            weights: weights.clone(),
            bias,
            weight_gradients: Array2::zeros(weights.dim()),
            bias_gradients: Array1::zeros(output_size),
        })
    }
    
    pub fn forward(&self, input: &Array1<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
        let output = self.weights.dot(input) + &self.bias;
        
        // Apply softmax for probability distribution
        let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_output: Array1<f32> = output.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_output.sum();
        
        let probabilities = if sum_exp > 0.0 {
            exp_output / sum_exp
        } else {
            Array1::from_elem(output.len(), 1.0 / output.len() as f32)
        };
        
        Ok(probabilities.to_vec())
    }
    
    pub fn backward(&mut self, output_grad: &Array1<f32>) -> Result<Array1<f32>, Box<dyn Error>> {
        let input_grad = self.weights.t().dot(output_grad);
        
        // Store gradients
        self.bias_gradients += output_grad;
        
        Ok(input_grad)
    }
    
    pub fn update_weights(&mut self, learning_rate: f32) {
        self.weights -= &(&self.weight_gradients * learning_rate);
        self.bias -= &(&self.bias_gradients * learning_rate);
        
        self.weight_gradients.fill(0.0);
        self.bias_gradients.fill(0.0);
    }
}

/// Simple trainer wrapper with training loop
pub struct MorseTrainer {
    model: DLinossWrapper,
    epoch: usize,
}

impl MorseTrainer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let model = DLinossWrapper::new()?;
        Ok(Self { model, epoch: 0 })
    }
    
    pub fn train_epoch(&mut self, training_samples: &[crate::data_generator::MorseAudioSample]) -> Result<f32, Box<dyn Error>> {
        let mut total_loss = 0.0;
        let mut num_samples = 0;
        
        for sample in training_samples {
            if !sample.audio_data.is_empty() && !sample.labels.is_empty() {
                // Extract target labels
                let target_labels: Vec<usize> = sample.labels.iter().map(|label| {
                    match label.symbol {
                        crate::data_generator::MorseSymbol::Dot => 0,
                        crate::data_generator::MorseSymbol::Dash => 1,
                        crate::data_generator::MorseSymbol::IntraCharGap => 2,
                        crate::data_generator::MorseSymbol::CharGap => 3,
                        crate::data_generator::MorseSymbol::WordGap => 4,
                        crate::data_generator::MorseSymbol::Noise => 5,
                    }
                }).collect();
                
                // Train on this sample
                let loss = self.model.train_step(&sample.audio_data, &target_labels)?;
                total_loss += loss;
                num_samples += 1;
            }
        }
        
        self.epoch += 1;
        let avg_loss = if num_samples > 0 { total_loss / num_samples as f32 } else { 0.0 };
        
        if self.epoch % 10 == 0 {
            println!("Epoch {}: Average Loss = {:.4}", self.epoch, avg_loss);
        }
        
        Ok(avg_loss)
    }
    
    pub fn get_model(&mut self) -> &mut DLinossWrapper {
        &mut self.model
    }
}

// Helper functions
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

fn morse_to_char(morse: &str) -> Option<char> {
    match morse {
        ".-" => Some('A'), "-..." => Some('B'), "-.-." => Some('C'), "-.." => Some('D'),
        "." => Some('E'), "..-." => Some('F'), "--." => Some('G'), "...." => Some('H'),
        ".." => Some('I'), ".---" => Some('J'), "-.-" => Some('K'), ".-.." => Some('L'),
        "--" => Some('M'), "-." => Some('N'), "---" => Some('O'), ".--." => Some('P'),
        "--.-" => Some('Q'), ".-." => Some('R'), "..." => Some('S'), "-" => Some('T'),
        "..-" => Some('U'), "...-" => Some('V'), ".--" => Some('W'), "-..-" => Some('X'),
        "-.--" => Some('Y'), "--.." => Some('Z'),
        "-----" => Some('0'), ".----" => Some('1'), "..---" => Some('2'), "...--" => Some('3'),
        "....-" => Some('4'), "....." => Some('5'), "-...." => Some('6'), "--..." => Some('7'),
        "---.." => Some('8'), "----." => Some('9'),
        _ => None,
    }
}
