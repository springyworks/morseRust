


// Morse Code Audio Test Binary with DLinOSS AI Integration
// Real-time audio input/output with PipeWire and AI-powered decoding

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
// use std::time::Instant;
// use std::io::Write;
// use std::sync::mpsc;
// use cpal::{Device, Stream, StreamConfig, SampleFormat, SampleRate, traits::*};
// use crossbeam_channel::{bounded, Receiver, Sender};
use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::Read;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};


mod data_generator;
mod neural_network;
mod inference;

use data_generator::MorseDataGenerator;
use neural_network::{DLinossWrapper, MorseTrainer};

/// Morse code patterns mapping
fn create_morse_map() -> HashMap<String, char> {
    let mut morse_map = HashMap::new();
    
    // Letters
    morse_map.insert(".-".to_string(), 'A');
    morse_map.insert("-...".to_string(), 'B');
    morse_map.insert("-.-.".to_string(), 'C');
    morse_map.insert("-..".to_string(), 'D');
    morse_map.insert(".".to_string(), 'E');
    morse_map.insert("..-.".to_string(), 'F');
    morse_map.insert("--.".to_string(), 'G');
    morse_map.insert("....".to_string(), 'H');
    morse_map.insert("..".to_string(), 'I');
    morse_map.insert(".---".to_string(), 'J');
    morse_map.insert("-.-".to_string(), 'K');
    morse_map.insert(".-..".to_string(), 'L');
    morse_map.insert("--".to_string(), 'M');
    morse_map.insert("-.".to_string(), 'N');
    morse_map.insert("---".to_string(), 'O');
    morse_map.insert(".--.".to_string(), 'P');
    morse_map.insert("--.-".to_string(), 'Q');
    morse_map.insert(".-.".to_string(), 'R');
    morse_map.insert("...".to_string(), 'S');
    morse_map.insert("-".to_string(), 'T');
    morse_map.insert("..-".to_string(), 'U');
    morse_map.insert("...-".to_string(), 'V');
    morse_map.insert(".--".to_string(), 'W');
    morse_map.insert("-..-".to_string(), 'X');
    morse_map.insert("-.--".to_string(), 'Y');
    morse_map.insert("--..".to_string(), 'Z');
    
    // Numbers
    morse_map.insert("-----".to_string(), '0');
    morse_map.insert(".----".to_string(), '1');
    morse_map.insert("..---".to_string(), '2');
    morse_map.insert("...--".to_string(), '3');
    morse_map.insert("....-".to_string(), '4');
    morse_map.insert(".....".to_string(), '5');
    morse_map.insert("-....".to_string(), '6');
    morse_map.insert("--...".to_string(), '7');
    morse_map.insert("---..".to_string(), '8');
    morse_map.insert("----.".to_string(), '9');
    
    morse_map
}

/// DLinOSS (Damped Linear Oscillators) Architecture for Morse Code
/// 
/// For Morse code audio decoding, we need:
/// 1. Frequency analysis (tone detection)
/// 2. Temporal pattern recognition (dots vs dashes, timing)
/// 3. Noise filtering and signal conditioning
///
/// Proposed Architecture:
/// - Input: Raw audio samples (44.1kHz, mono)
/// - Layer 1: 3 DLinOSS layers for frequency decomposition (low, mid, high freq)
/// - Layer 2: 2 DLinOSS layers for temporal pattern extraction 
/// - Layer 3: 1 DLinOSS layer for noise filtering
/// - Activation: GELU for smooth gradients, ReLU for sparse features
/// - Output: Dot/Dash/Silence classification + timing confidence
#[derive(Debug, Clone)]
pub struct DLinOSSMorseDecoder {
    // Frequency analysis layers (3 layers)
    freq_layers: Vec<DLinOSSLayer>,
    // Temporal pattern layers (2 layers) 
    temporal_layers: Vec<DLinOSSLayer>,
    // Noise filtering layer (1 layer)
    noise_filter: DLinOSSLayer,
    // Classification head
    classifier: LinearLayer,
    // Audio buffer for real-time processing
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    // Decoded morse pattern
    morse_pattern: Arc<Mutex<String>>,
    // Confidence threshold
    confidence_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct DLinOSSLayer {
    // Damped Linear Oscillator parameters
    damping_factor: f32,
    resonant_frequency: f32,
    amplitude: f32,
    phase: f32,
    // Learnable parameters
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    // Activation function
    activation: ActivationType,
}

#[derive(Debug, Clone)]
pub enum ActivationType {
    GELU,    // For smooth frequency analysis
    ReLU,    // For sparse temporal features
    Sigmoid, // For confidence outputs
}

#[derive(Debug, Clone)]
pub struct LinearLayer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

impl DLinOSSMorseDecoder {
    pub fn new() -> Self {
        // Architecture: 6 total DLinOSS layers + classification
        // - 3 frequency analysis layers (GELU activation)
        // - 2 temporal pattern layers (ReLU activation) 
        // - 1 noise filter layer (GELU activation)
        
        let mut freq_layers = Vec::new();
        
        // Frequency analysis layers - each tuned to different frequency ranges
        freq_layers.push(DLinOSSLayer::new(256, 128, 600.0, ActivationType::GELU)); // Low freq (600Hz - typical morse tone)
        freq_layers.push(DLinOSSLayer::new(128, 64, 800.0, ActivationType::GELU));  // Mid freq  
        freq_layers.push(DLinOSSLayer::new(64, 32, 1000.0, ActivationType::GELU)); // High freq
        
        let mut temporal_layers = Vec::new();
        
        // Temporal pattern recognition layers - sparse feature extraction
        temporal_layers.push(DLinOSSLayer::new(96, 48, 10.0, ActivationType::ReLU)); // Combined freq features -> temporal patterns
        temporal_layers.push(DLinOSSLayer::new(48, 24, 5.0, ActivationType::ReLU));  // Pattern refinement
        
        // Noise filtering layer - smooth denoising
        let noise_filter = DLinOSSLayer::new(24, 12, 2.0, ActivationType::GELU);
        
        // Classification layer: 12 -> 4 outputs (Dot, Dash, Silence, Confidence)
        let classifier = LinearLayer::new(12, 4);
        
        Self {
            freq_layers,
            temporal_layers,
            noise_filter,
            classifier,
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            morse_pattern: Arc::new(Mutex::new(String::new())),
            confidence_threshold: 0.7,
        }
    }
    
    /// Process audio samples and decode morse patterns
    pub fn process_audio(&mut self, samples: &[f32]) -> Option<MorseEvent> {
        // 1. Frequency analysis (3 DLinOSS layers with GELU)
        let mut freq_features = Vec::new();
        for layer in &mut self.freq_layers {
            let features = layer.forward(samples);
            freq_features.extend(features);
        }
        
        // 2. Temporal pattern extraction (2 DLinOSS layers with ReLU)
        let mut temporal_features = freq_features.clone();
        for layer in &mut self.temporal_layers {
            temporal_features = layer.forward(&temporal_features);
        }
        
        // 3. Noise filtering (1 DLinOSS layer with GELU)
        let filtered = self.noise_filter.forward(&temporal_features);
        
        // 4. Classification
        let output = self.classifier.forward(&filtered);
        
        // 5. Decode results
        self.decode_output(&output)
    }
    
    
    fn decode_output(&self, output: &[f32]) -> Option<MorseEvent> {
        if output.len() < 4 {
            return None;
        }
        
        // [dot, dash, space, noise] probabilities
        let max_idx = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?
            .0;
        
        let confidence = output[max_idx];
        
        match max_idx {
            0 => Some(MorseEvent::Dot(confidence)),
            1 => Some(MorseEvent::Dash(confidence)),
            2 => Some(MorseEvent::Silence(confidence)),
            _ => None,
        }
    }
}

impl DLinOSSLayer {
    pub fn new(input_size: usize, output_size: usize, freq: f32, activation: ActivationType) -> Self {
        // Initialize weights with small random values scaled by frequency
        let mut weights = Vec::new();
        for _ in 0..output_size {
            let mut row = Vec::new();
            for _ in 0..input_size {
                let weight = (rand::random::<f32>() - 0.5) * 0.1 * (freq / 1000.0);
                row.push(weight);
            }
            weights.push(row);
        }
        
        // Frequency-tuned bias initialization
        let bias: Vec<f32> = (0..output_size).map(|i| {
            let bias_factor = (i as f32 / output_size as f32) * 0.1 - 0.05;
            bias_factor * (freq / 1000.0) // Scale by frequency
        }).collect();
        
        Self {
            damping_factor: 0.05 + (freq / 10000.0), // Frequency-dependent damping
            resonant_frequency: freq,
            amplitude: 1.0,
            phase: 0.0,
            weights,
            bias,
            activation,
        }
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.weights.len()];
        
        // DLinOSS computation with damped oscillation
        for (i, weight_row) in self.weights.iter().enumerate() {
            let mut sum = 0.0;
            
            for (j, &input_val) in input.iter().enumerate() {
                if j < weight_row.len() {
                    // Apply damped linear oscillation
                    let oscillation = (self.resonant_frequency * input_val + self.phase).sin() 
                                    * (-self.damping_factor * input_val.abs()).exp();
                    
                    sum += weight_row[j] * input_val * oscillation;
                }
            }
            
            sum += self.bias[i];
            
            // Apply activation function
            output[i] = match self.activation {
                ActivationType::GELU => gelu(sum),
                ActivationType::ReLU => sum.max(0.0),
                ActivationType::Sigmoid => 1.0 / (1.0 + (-sum).exp()),
            };
        }
        
        output
    }
}

impl LinearLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut weights = Vec::new();
        for _ in 0..output_size {
            let mut row = Vec::new();
            for _ in 0..input_size {
                row.push((rand::random::<f32>() - 0.5) * 0.1);
            }
            weights.push(row);
        }
        
        let bias = vec![0.0; output_size];
        
        Self { weights, bias }
    }
    
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.weights.len()];
        
        for (i, weight_row) in self.weights.iter().enumerate() {
            let mut sum = 0.0;
            for (j, &input_val) in input.iter().enumerate() {
                if j < weight_row.len() {
                    sum += weight_row[j] * input_val;
                }
            }
            sum += self.bias[i];
            output[i] = sum;
        }
        
        output
    }
}

// GELU activation function
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

#[derive(Debug, Clone)]
pub enum MorseEvent {
    Dot(f32),      // confidence
    Dash(f32),     // confidence
    Silence(f32),  // confidence
}

// Placeholder for external crate functions until we add dependencies
mod rand {
    pub fn random<T>() -> T 
    where 
        T: From<u8>
    {
        T::from(42) // Placeholder
    }
}

#[derive(Parser)]
#[command(name = "morseRust")]
#[command(about = "A professional morse code audio and AI system")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Frequency for morse tone generation (Hz)
    #[arg(short, long, default_value = "800")]
    frequency: f32,
    
    /// Words per minute for morse generation
    #[arg(short, long, default_value = "20")]
    wpm: u32,
}

#[derive(Subcommand)]
enum Commands {
    /// Test morse code mapping
    Test {
        /// Test specific patterns
        #[arg(short, long)]
        pattern: Option<String>,
    },
    /// Generate morse code audio for text
    Generate {
        /// Text to convert to morse code
        text: String,
        /// Output mode (standard, realistic, musical, multifreq)
        #[arg(short, long, default_value = "standard")]
        mode: String,
    },
    /// Decode morse code from audio input
    Decode {
        /// Input audio file (future feature)
        #[arg(short, long)]
        input: Option<String>,
    },
    /// Play a .bin file as audio
    PlayBin {
        /// Path to .bin file
        file: String,
        /// Sample rate (default 44100)
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,
    },
    /// Play a range or list of .bin files as audio (inclusive, hardcoded naming)
    PlayRange {
        /// Range/list in the form 0-2,4,5,6-7 or 1 2 (comma/space/range supported)
        range: String,
        /// Sample rate (default 44100)
        #[arg(short, long, default_value = "44100")]
        sample_rate: u32,
    },
    /// Train the AI decoder
    Train {
        /// Training data file (future feature)
        #[arg(short, long)]
        data: Option<String>,
        /// Number of epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,
    },
    /// Generate training data
    GenerateData {
        /// Number of samples to generate
        #[arg(short, long, default_value = "1000")]
        num_samples: usize,
        /// Output directory
        #[arg(short, long, default_value = "./data")]
        output_dir: String,
    },
    /// Run inference on a dataset
    Inference {
        /// Data directory for inference
        #[arg(short, long, default_value = "data")]
        data_dir: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    if cli.verbose {
        println!("🔊 Morse Code System - Verbose Mode");
    } else {
        println!("🔊 Morse Code System");
    }
    println!("==================");
    
    match &cli.command {
        Some(Commands::Test { pattern }) => {
            test_morse_patterns(pattern.as_deref());
        },
        Some(Commands::Generate { text, mode }) => {
            let audio = generate_morse_audio_with_data(text, mode, cli.frequency, cli.wpm)?;
            println!("▶️  Playing generated Morse audio...");
            play_audio(&audio, 44100)?;
        },
        Some(Commands::Decode { input }) => {
            decode_morse_audio(input.as_deref())?;
        },
        Some(Commands::PlayBin { file, sample_rate }) => {
            let mut f = File::open(file)?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            // Convert bytes to f32
            let audio: Vec<f32> = buf.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            println!("▶️  Playing {} ({} samples)...", file, audio.len());
            play_audio(&audio, *sample_rate)?;
        },
        Some(Commands::PlayRange { range, sample_rate }) => {
            // Accepts: 0-2,4,5,6-7 or 1 2 or 1,2 etc.
            let mut indices = std::collections::BTreeSet::new();
            for token in range.replace(',', " ").split_whitespace() {
                if let Some(dash) = token.find('-') {
                    let (start, end) = token.split_at(dash);
                    let start = start.trim().parse::<usize>();
                    let end = end[1..].trim().parse::<usize>();
                    match (start, end) {
                        (Ok(s), Ok(e)) if s <= e => {
                            for i in s..=e {
                                indices.insert(i);
                            }
                        },
                        _ => {
                            eprintln!("Invalid range: {}", token);
                        }
                    }
                } else {
                    match token.trim().parse::<usize>() {
                        Ok(i) => { indices.insert(i); },
                        Err(_) => eprintln!("Invalid sample number: {}", token),
                    }
                }
            }
            if indices.is_empty() {
                eprintln!("No valid sample indices specified.");
                return Ok(());
            }
            let mut playable = false;
            for i in &indices {
                let file = format!("data/sample_{:06}.bin", i);
                if !std::path::Path::new(&file).exists() {
                    eprintln!("Warning: file does not exist: {}", file);
                }
            }
            for i in indices {
                let file = format!("data/sample_{:06}.bin", i);
                let json_file = format!("data/sample_{:06}.json", i);
                let mut snr_str = String::from("?");
                let mut text_str = String::from("?");
                if let Ok(json_content) = std::fs::read_to_string(&json_file) {
                    if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&json_content) {
                        if let Some(metadata) = json_val.get("metadata") {
                            if let Some(snr) = metadata.get("snr_db").and_then(|v| v.as_f64()) {
                                snr_str = format!("{:.2}", snr);
                            }
                            if let Some(txt) = metadata.get("text").and_then(|v| v.as_str()) {
                                text_str = txt.to_string();
                            }
                        }
                    }
                }
                if let Ok(mut f) = File::open(&file) {
                    let mut buf = Vec::new();
                    f.read_to_end(&mut buf)?;
                    let audio: Vec<f32> = buf.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    println!("▶️  Playing {} ({} samples) [SNR: {} dB, Text: {}]...", file, audio.len(), snr_str, text_str);
                    play_audio(&audio, *sample_rate)?;
                    playable = true;
                }
            }
            if !playable {
                eprintln!("No playable files found for the specified range/list.");
            }
        },
        Some(Commands::Train { data, epochs }) => {
            train_ai_decoder(data.as_deref(), *epochs)?;
        },
        Some(Commands::Inference { data_dir }) => {
            use crate::inference::run_inference;
            let mut model = DLinossWrapper::new()?;
            run_inference(&mut model, data_dir)?;
        },
        Some(Commands::GenerateData { num_samples, output_dir }) => {
            generate_training_data(*num_samples, output_dir, cli.frequency)?;
        },
        None => {
            // Default behavior - run tests
            test_morse_patterns(None);
        }
    }
    
    Ok(())
}

fn test_morse_patterns(specific_pattern: Option<&str>) {
    let morse_map = create_morse_map();
    println!("Morse map loaded with {} entries", morse_map.len());
    
    let test_patterns = if let Some(pattern) = specific_pattern {
        vec![(pattern, "?")]
    } else {
        vec![
            (".-", "A"),
            ("-...", "B"), 
            ("-.-.", "C"),
            ("...", "S"),
            ("---", "O"),
            ("...---...", "SOS"),
        ]
    };
    
    for (pattern, expected) in test_patterns {
        if let Some(&decoded) = morse_map.get(pattern) {
            println!("  {} -> {} ✓ (expected {})", pattern, decoded, expected);
        } else {
            println!("  {} -> ? ✗ (expected {})", pattern, expected);
        }
    }
    
    println!("\n🎯 DLinOSS AI Decoder initialized");
    let _decoder = DLinOSSMorseDecoder::new();
    println!("✅ System ready!");
}


fn generate_morse_audio_with_data(text: &str, mode: &str, frequency: f32, wpm: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    println!("🎵 Generating morse audio for: '{}'", text);
    println!("   Mode: {}, Frequency: {} Hz, WPM: {}", mode, frequency, wpm);
    // Use MorseDataGenerator to synthesize audio
    let generator = MorseDataGenerator::new(44100, frequency);
    let sample = generator.generate_morse_sample(text, wpm, frequency, 0.01)?;
    println!("✅ Audio generated ({} samples)", sample.audio_data.len());
    Ok(sample.audio_data)
}

fn play_audio(audio: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("No output device available")?;
    let config = device.default_output_config()?;
    let sample_format = config.sample_format();
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };
    let audio_data = Arc::new(audio.to_vec());
    let total = audio_data.len();
    let _idx = 0;
    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let audio_data = Arc::clone(&audio_data);
            let mut idx = 0;
            device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    for sample in data.iter_mut() {
                        *sample = if idx < total { audio_data[idx] } else { 0.0 };
                        idx += 1;
                    }
                },
                |_| {},
                None,
            )?
        },
        cpal::SampleFormat::I16 => {
            let audio_data = Arc::clone(&audio_data);
            let mut idx = 0;
            device.build_output_stream(
                &config,
                move |data: &mut [i16], _| {
                    for sample in data.iter_mut() {
                        let val = if idx < total { audio_data[idx] } else { 0.0 };
                        *sample = (val * i16::MAX as f32) as i16;
                        idx += 1;
                    }
                },
                |_| {},
                None,
            )?
        },
        cpal::SampleFormat::U16 => {
            let audio_data = Arc::clone(&audio_data);
            let mut idx = 0;
            device.build_output_stream(
                &config,
                move |data: &mut [u16], _| {
                    for sample in data.iter_mut() {
                        let val = if idx < total { audio_data[idx] } else { 0.0 };
                        *sample = ((val * 0.5 + 0.5) * u16::MAX as f32) as u16;
                        idx += 1;
                    }
                },
                |_| {},
                None,
            )?
        },
        _ => return Err("Unsupported sample format".into()),
    };
    stream.play()?;
    // Wait for audio to finish
    std::thread::sleep(std::time::Duration::from_secs_f32(audio_data.len() as f32 / sample_rate as f32 + 0.5));
    Ok(())
}

fn decode_morse_audio(input_file: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎧 Decoding morse audio");
    let mut network = DLinossWrapper::new().expect("Failed to create D-LinOSS network");
    if let Some(file) = input_file {
        println!("   Input file: {}", file);
        // TODO: Load and preprocess audio file into feature vector
        let dummy_samples = vec![0.5, 0.3, 0.1, 0.0, 0.2];
        let output = network.forward(&dummy_samples);
        println!("   D-LinOSS output: {:?}", output);
        println!("✅ Decoding completed");
        Ok(())
    } else {
        decode_morse_from_microphone()
    }
}

fn decode_morse_from_microphone() -> Result<(), Box<dyn std::error::Error>> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;
    println!("🎤 Listening to microphone (PipeWire/CPAL)... Speak Morse now!");
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No input device available")?;
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buffer_clone = buffer.clone();
    let print_interval = std::time::Duration::from_millis(1000);
    let mut network = DLinossWrapper::new().expect("Failed to create D-LinOSS network");
    // let mut last_print = Instant::now();

    #[cfg(feature = "mic_level_test")]
    println!("[mic_level_test] Printing audio level only. No Morse decoding.");

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                let mut buf = buffer_clone.lock().unwrap();
                // Downmix to mono if needed
                if channels == 1 {
                    buf.extend_from_slice(data);
                } else {
                    for frame in data.chunks(channels) {
                        buf.push(frame[0]);
                    }
                }
            },
            |err| eprintln!("Input stream error: {:?}", err),
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _| {
                let mut buf = buffer_clone.lock().unwrap();
                if channels == 1 {
                    buf.extend(data.iter().map(|&x| x as f32 / i16::MAX as f32));
                } else {
                    for frame in data.chunks(channels) {
                        buf.push(frame[0] as f32 / i16::MAX as f32);
                    }
                }
            },
            |err| eprintln!("Input stream error: {:?}", err),
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config.into(),
            move |data: &[u16], _| {
                let mut buf = buffer_clone.lock().unwrap();
                if channels == 1 {
                    buf.extend(data.iter().map(|&x| x as f32 / u16::MAX as f32 - 0.5));
                } else {
                    for frame in data.chunks(channels) {
                        buf.push(frame[0] as f32 / u16::MAX as f32 - 0.5);
                    }
                }
            },
            |err| eprintln!("Input stream error: {:?}", err),
            None,
        )?,
        _ => return Err("Unsupported input sample format".into()),
    };
    stream.play()?;
    println!("[Press Ctrl+C to stop]");
    #[cfg(feature = "mic_level_test")]
    {
        // Improved AGC: noise floor tracking, hang timer, min/max gain
        let mut sound_active = false;
        let mut sound_len = 0;
        let mut silence_len = 0;
        let mut agc_level = 0.05f32; // Initial AGC target RMS
        let mut noise_floor = 0.01f32; // Track background noise
        let mut agc_hang = 0u32; // Hang timer for AGC
        let agc_decay = 0.997f32; // AGC smoothing factor (slower)
        let agc_attack = 0.7f32; // AGC fast attack for loud sounds
        let agc_hang_time = 10u32; // Number of 50ms frames to hang after sound (0.5s)
        let min_agc = 0.01f32;
        let max_agc = 0.5f32;
        let base_threshold = 0.22f32; // Slightly higher threshold
        let dot_max = (sample_rate as f32 * 0.18) as usize; // ~180ms
        let dash_min = (sample_rate as f32 * 0.18) as usize; // >180ms
        let dash_max = (sample_rate as f32 * 0.5) as usize;  // <500ms
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let mut buf = buffer.lock().unwrap();
            if buf.len() >= (sample_rate as usize / 20) { // 50ms
                let samples: Vec<f32> = buf.drain(..(sample_rate as usize / 20)).collect();
                drop(buf);
                let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
                // Track noise floor (moving average, only when not sounding)
                if !sound_active && rms < agc_level * 0.7 {
                    noise_floor = 0.98 * noise_floor + 0.02 * rms;
                }
                // AGC: adapt only when not sounding or after hang
                if !sound_active && agc_hang == 0 {
                    if rms > agc_level {
                        agc_level = agc_attack * rms + (1.0 - agc_attack) * agc_level;
                    } else {
                        agc_level = agc_decay * agc_level + (1.0 - agc_decay) * rms;
                    }
                    agc_level = agc_level.clamp(min_agc.max(noise_floor * 2.0), max_agc);
                }
                let threshold = base_threshold * agc_level.max(min_agc);
                if rms > threshold {
                    sound_len += samples.len();
                    silence_len = 0;
                    if !sound_active {
                        sound_active = true;
                        agc_hang = agc_hang_time; // Hold AGC for a while after sound
                    }
                } else {
                    if sound_active {
                        // End of sound, decide dot or dash
                        if sound_len > 0 && sound_len <= dot_max {
                            print!("•");
                            std::io::Write::flush(&mut std::io::stdout()).ok();
                        } else if sound_len > dash_min && sound_len <= dash_max {
                            print!("━");
                            std::io::Write::flush(&mut std::io::stdout()).ok();
                        }
                        sound_active = false;
                        sound_len = 0;
                    }
                    silence_len += samples.len();
                    // Optionally print space after long silence
                    if silence_len > dash_max {
                        print!(" ");
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                        silence_len = 0;
                    }
                    // AGC hang timer
                    if agc_hang > 0 {
                        agc_hang -= 1;
                    }
                }
            }
        }
    }
    #[cfg(not(feature = "mic_level_test"))]
    loop {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut buf = buffer.lock().unwrap();
        if buf.len() >= sample_rate as usize {
            // Take 1 second of audio
            let samples: Vec<f32> = buf.drain(..sample_rate as usize).collect();
            drop(buf);
            let output = network.forward(&samples);
            println!("   D-LinOSS output: {:?}", output);
        }
    }
}

fn train_ai_decoder(data_file: Option<&str>, epochs: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Training Advanced DLinOSS Morse Decoder");
    if let Some(file) = data_file {
        println!("   Training data: {}", file);
    } else {
        println!("   Using default training data from data/");
    }
    
    println!("   Epochs: {}", epochs);
    println!("   Architecture: Multi-layer DLinOSS + Temporal Processing");
    println!("   - 4 DLinOSS frequency/pattern analysis layers");
    println!("   - 1 LSTM-like temporal processor");
    println!("   - 1 classification head with 6 classes");
    
    // Load training data
    let data_dir = data_file.unwrap_or("data");
    let training_samples = load_training_data(data_dir)?;
    
    if training_samples.is_empty() {
        println!("   ❌ No training data found. Generate data first with: cargo run -- generate-data");
        return Ok(());
    }
    
    println!("   📊 Loaded {} training samples", training_samples.len());
    
    // Split into training and validation
    let split_idx = (training_samples.len() as f32 * 0.8) as usize;
    let (train_samples, val_samples) = training_samples.split_at(split_idx);
    
    println!("   📈 Training samples: {}", train_samples.len());
    println!("   📉 Validation samples: {}", val_samples.len());
    
    // Create and train the network
    let mut trainer = MorseTrainer::new()?;
    
    println!("\n🚀 Starting training...");
    let mut best_val_loss = f32::INFINITY;
    let mut patience_counter = 0;
    let patience = 10; // Early stopping patience
    
    for epoch in 1..=epochs {
        // Training phase
        let train_loss = trainer.train_epoch(train_samples)?;
        
        // Validation phase (simplified - just average loss)
        let mut val_loss = 0.0;
        let mut val_count = 0;
        
        for sample in val_samples.iter().take(10) { // Limit validation samples for speed
            if !sample.audio_data.is_empty() && !sample.labels.is_empty() {
                let target_labels: Vec<usize> = sample.labels.iter().map(|label| {
                    match label.symbol {
                        data_generator::MorseSymbol::Dot => 0,
                        data_generator::MorseSymbol::Dash => 1,
                        data_generator::MorseSymbol::IntraCharGap => 2,
                        data_generator::MorseSymbol::CharGap => 3,
                        data_generator::MorseSymbol::WordGap => 4,
                        data_generator::MorseSymbol::Noise => 5,
                    }
                }).collect();
                
                // Forward pass only (no training)
                let prediction = trainer.get_model().forward(&sample.audio_data)?;
                
                // Calculate validation loss
                let mut sample_loss = 0.0;
                for i in 0..prediction.len().min(6) {
                    let target_prob = if target_labels.contains(&i) { 1.0 } else { 0.0 };
                    sample_loss -= target_prob * prediction[i].max(1e-15).ln();
                }
                
                val_loss += sample_loss;
                val_count += 1;
            }
        }
        
        val_loss /= val_count as f32;
        
        // Print progress
        if epoch % 5 == 0 || epoch <= 10 {
            println!("   Epoch {}/{}: Train Loss = {:.4}, Val Loss = {:.4}", 
                    epoch, epochs, train_loss, val_loss);
        }
        
        // Early stopping check
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }
        
        if patience_counter >= patience {
            println!("   🛑 Early stopping triggered at epoch {}", epoch);
            break;
        }
        
        // Test inference on a sample every 20 epochs
        if epoch % 20 == 0 && !train_samples.is_empty() {
            let test_sample = &train_samples[0];
            let decoded = trainer.get_model().decode_sequence(&test_sample.audio_data)?;
            println!("   🧪 Test decode: \"{}\" -> \"{}\"", test_sample.metadata.text, decoded);
        }
    }
    
    println!("\n✅ Training completed!");
    println!("   🏆 Best validation loss: {:.4}", best_val_loss);
    
    // Final evaluation
    println!("\n📊 Final Evaluation:");
    let mut correct_predictions = 0;
    let mut total_predictions = 0;
    
    for sample in val_samples.iter().take(20) {
        if !sample.audio_data.is_empty() {
            let prediction = trainer.get_model().forward(&sample.audio_data)?;
            let _predicted_symbol = trainer.get_model().predict_symbol(&prediction);
            let decoded_text = trainer.get_model().decode_sequence(&sample.audio_data)?;
            
            // Simple accuracy check based on character matching
            let accuracy = calculate_text_similarity(&decoded_text, &sample.metadata.text);
            if accuracy > 0.5 { // More than 50% similarity
                correct_predictions += 1;
            }
            total_predictions += 1;
            
            if total_predictions <= 5 { // Show first 5 examples
                println!("   Sample \"{}\": Decoded \"{}\" (Accuracy: {:.1}%)", 
                        sample.metadata.text, decoded_text, accuracy * 100.0);
            }
        }
    }
    
    if total_predictions > 0 {
        let accuracy = correct_predictions as f32 / total_predictions as f32;
        println!("   🎯 Overall Accuracy: {:.1}% ({}/{})", 
                accuracy * 100.0, correct_predictions, total_predictions);
    }
    
    Ok(())
}

fn calculate_text_similarity(text1: &str, text2: &str) -> f32 {
    let chars1: Vec<char> = text1.chars().collect();
    let chars2: Vec<char> = text2.chars().collect();
    
    if chars1.is_empty() && chars2.is_empty() {
        return 1.0;
    }
    
    let max_len = chars1.len().max(chars2.len());
    if max_len == 0 {
        return 1.0;
    }
    
    let mut matching = 0;
    for i in 0..max_len {
        let c1 = chars1.get(i).copied().unwrap_or(' ');
        let c2 = chars2.get(i).copied().unwrap_or(' ');
        if c1 == c2 {
            matching += 1;
        }
    }
    
    matching as f32 / max_len as f32
}

fn generate_training_data(num_samples: usize, output_dir: &str, base_frequency: f32) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 Generating {} training samples", num_samples);
    println!("   Output directory: {}", output_dir);
    println!("   Base frequency: {} Hz", base_frequency);
    
    let generator = MorseDataGenerator::new(44100, base_frequency);
    generator.generate_training_dataset(output_dir, num_samples)?;
    
    println!("✅ Training data generation completed");
    Ok(())
}

fn load_training_data(data_dir: &str) -> Result<Vec<data_generator::MorseAudioSample>, Box<dyn std::error::Error>> {
    use std::fs;
    use std::path::Path;
    
    let mut samples = Vec::new();
    
    if !Path::new(data_dir).exists() {
        return Ok(samples);
    }
    
    // Load JSON files (labels) and corresponding binary files (audio)
    for entry in fs::read_dir(data_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Some(_stem) = path.file_stem().and_then(|s| s.to_str()) {
                let json_content = fs::read_to_string(&path)?;
                // Deserialize only labels and metadata, ignore audio_data
                let json_value: serde_json::Value = serde_json::from_str(&json_content)?;
                let labels = serde_json::from_value(json_value["labels"].clone())?;
                let metadata = serde_json::from_value(json_value["metadata"].clone())?;
                let mut sample = data_generator::MorseAudioSample {
                    audio_data: Vec::new(),
                    labels,
                    metadata,
                };
                // Load corresponding binary audio data
                let binary_path = path.with_extension("bin");
                if binary_path.exists() {
                    let binary_data = fs::read(&binary_path)?;
                    let mut audio_data = Vec::new();
                    for chunk in binary_data.chunks_exact(4) {
                        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        audio_data.push(value);
                    }
                    sample.audio_data = audio_data;
                    samples.push(sample);
                }
            }
        }
    }
    
    Ok(samples)
}
