// inference.rs
// Advanced inference logic for Morse code neural network with proper decoding

use crate::data_generator::MorseAudioSample;
use crate::neural_network::DLinossWrapper;
use std::fs;
use std::collections::HashMap;

pub fn run_inference(model: &mut DLinossWrapper, data_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Running advanced Morse code inference...");
    
    let mut samples = load_inference_samples(data_dir)?;
    
    if samples.is_empty() {
        println!("❌ No samples found in {}", data_dir);
        return Ok(());
    }
    
    // Limit to first 10 samples for demo
    samples.truncate(10);
    
    let mut total_accuracy = 0.0;
    let mut symbol_accuracy = HashMap::new();
    let mut confusion_matrix: HashMap<String, HashMap<String, usize>> = HashMap::new();
    
    println!("📊 Processing {} samples...\n", samples.len());
    
    for (i, sample) in samples.iter().enumerate() {
        println!("Sample {}: \"{}\" (SNR: {:.1} dB)", 
                i + 1, 
                sample.metadata.text, 
                sample.metadata.snr_db);
        
        // Run inference on the sample
        let prediction = model.forward(&sample.audio_data)?;
        let predicted_symbol = model.predict_symbol(&prediction);
        
        // Decode the entire sequence
        let decoded_text = model.decode_sequence(&sample.audio_data)?;
        
        // Analyze predictions vs ground truth
        let (sample_accuracy, symbol_stats) = analyze_prediction(&sample, &prediction, &decoded_text);
        
        total_accuracy += sample_accuracy;
        
        // Update symbol accuracy tracking
        for (symbol, (correct, total)) in symbol_stats {
            let entry = symbol_accuracy.entry(symbol).or_insert((0, 0));
            entry.0 += correct;
            entry.1 += total;
        }
        
        // Print results
        println!("  Prediction: {} (conf: {:.3})", predicted_symbol, prediction.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0));
        println!("  Decoded: \"{}\"", decoded_text);
        println!("  Accuracy: {:.1}%", sample_accuracy * 100.0);
        
        // Show detailed symbol breakdown
        if !sample.labels.is_empty() {
            print!("  Ground truth symbols: ");
            for label in &sample.labels {
                let symbol_name = match label.symbol {
                    crate::data_generator::MorseSymbol::Dot => "•",
                    crate::data_generator::MorseSymbol::Dash => "━",
                    crate::data_generator::MorseSymbol::IntraCharGap => "·",
                    crate::data_generator::MorseSymbol::CharGap => " ",
                    crate::data_generator::MorseSymbol::WordGap => " / ",
                    crate::data_generator::MorseSymbol::Noise => "~",
                };
                print!("{}", symbol_name);
            }
            println!();
        }
        
        println!();
    }
    
    // Print overall statistics
    let avg_accuracy = total_accuracy / samples.len() as f32;
    println!("📈 Overall Results:");
    println!("  Average Accuracy: {:.1}%", avg_accuracy * 100.0);
    println!("  Total Samples: {}", samples.len());
    
    println!("\n🎯 Symbol-wise Accuracy:");
    for (symbol, (correct, total)) in symbol_accuracy {
        if total > 0 {
            let accuracy = correct as f32 / total as f32;
            println!("  {}: {:.1}% ({}/{})", symbol, accuracy * 100.0, correct, total);
        }
    }
    
    // Character-level accuracy analysis
    println!("\n📝 Character Decoding Analysis:");
    let mut char_correct = 0;
    let mut char_total = 0;
    
    for sample in &samples {
        let decoded = model.decode_sequence(&sample.audio_data)?;
        let ground_truth = &sample.metadata.text;
        
        let char_accuracy = calculate_character_accuracy(&decoded, ground_truth);
        char_correct += char_accuracy.0;
        char_total += char_accuracy.1;
    }
    
    if char_total > 0 {
        println!("  Character Accuracy: {:.1}% ({}/{})", 
                (char_correct as f32 / char_total as f32) * 100.0, 
                char_correct, 
                char_total);
    }
    
    println!("\n✅ Inference complete!");
    Ok(())
}

fn load_inference_samples(data_dir: &str) -> Result<Vec<MorseAudioSample>, Box<dyn std::error::Error>> {
    let mut samples = Vec::new();
    
    if !std::path::Path::new(data_dir).exists() {
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
                let mut sample = MorseAudioSample {
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
    
    // Sort by filename for consistent ordering
    samples.sort_by(|a, b| a.metadata.text.cmp(&b.metadata.text));
    
    Ok(samples)
}

fn analyze_prediction(
    sample: &MorseAudioSample, 
    prediction: &[f32], 
    decoded_text: &str
) -> (f32, HashMap<String, (usize, usize)>) {
    let mut symbol_stats = HashMap::new();
    
    // Analyze symbol-level accuracy
    if !sample.labels.is_empty() {
        // Get the most frequent symbol as ground truth
        let mut symbol_counts = HashMap::new();
        for label in &sample.labels {
            let symbol_name = match label.symbol {
                crate::data_generator::MorseSymbol::Dot => "Dot",
                crate::data_generator::MorseSymbol::Dash => "Dash",
                crate::data_generator::MorseSymbol::IntraCharGap => "IntraCharGap",
                crate::data_generator::MorseSymbol::CharGap => "CharGap",
                crate::data_generator::MorseSymbol::WordGap => "WordGap",
                crate::data_generator::MorseSymbol::Noise => "Noise",
            };
            *symbol_counts.entry(symbol_name.to_string()).or_insert(0) += 1;
        }
        
        // Find the most frequent ground truth symbol
        let ground_truth_symbol = symbol_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(symbol, _)| symbol.clone())
            .unwrap_or_else(|| "Unknown".to_string());
        
        // Get predicted symbol
        let max_idx = prediction.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let symbol_names = ["Dot", "Dash", "IntraCharGap", "CharGap", "WordGap", "Noise"];
        let predicted_symbol = symbol_names.get(max_idx).unwrap_or(&"Unknown").to_string();
        
        // Update symbol statistics
        let entry = symbol_stats.entry(ground_truth_symbol.clone()).or_insert((0, 0));
        entry.1 += 1; // Total count
        if predicted_symbol == ground_truth_symbol {
            entry.0 += 1; // Correct count
        }
    }
    
    // Calculate character-level accuracy
    let char_accuracy = calculate_character_accuracy(decoded_text, &sample.metadata.text);
    let sample_accuracy = if char_accuracy.1 > 0 {
        char_accuracy.0 as f32 / char_accuracy.1 as f32
    } else {
        0.0
    };
    
    (sample_accuracy, symbol_stats)
}

fn calculate_character_accuracy(predicted: &str, ground_truth: &str) -> (usize, usize) {
    let predicted_chars: Vec<char> = predicted.chars().collect();
    let truth_chars: Vec<char> = ground_truth.chars().collect();
    
    let mut correct = 0;
    let max_len = predicted_chars.len().max(truth_chars.len());
    
    for i in 0..max_len {
        let pred_char = predicted_chars.get(i).copied().unwrap_or(' ');
        let truth_char = truth_chars.get(i).copied().unwrap_or(' ');
        
        if pred_char == truth_char {
            correct += 1;
        }
    }
    
    (correct, max_len)
}

/// Advanced inference with confidence scoring
pub fn run_advanced_inference(model: &mut DLinossWrapper, data_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Running advanced Morse inference with confidence scoring...");
    
    let samples = load_inference_samples(data_dir)?;
    
    for (i, sample) in samples.iter().take(5).enumerate() {
        println!("\n🎵 Sample {}: \"{}\"", i + 1, sample.metadata.text);
        println!("  SNR: {:.1} dB, Frequency: {:.1} Hz, WPM: {}", 
                sample.metadata.snr_db, 
                sample.metadata.frequency, 
                sample.metadata.wpm);
        
        // Extract features and analyze
        let prediction = model.forward(&sample.audio_data)?;
        let decoded = model.decode_sequence(&sample.audio_data)?;
        
        // Calculate confidence metrics
        let max_confidence = prediction.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let entropy = calculate_entropy(&prediction);
        
        println!("  🎯 Decoded: \"{}\"", decoded);
        println!("  📊 Max Confidence: {:.3}", max_confidence);
        println!("  📈 Prediction Entropy: {:.3}", entropy);
        
        // Show prediction distribution
        let symbol_names = ["Dot", "Dash", "IntraGap", "CharGap", "WordGap", "Noise"];
        print!("  🔢 Distribution: ");
        for (i, &prob) in prediction.iter().enumerate() {
            if let Some(name) = symbol_names.get(i) {
                print!("{}:{:.2} ", name, prob);
            }
        }
        println!();
    }
    
    Ok(())
}

fn calculate_entropy(probabilities: &[f32]) -> f32 {
    let mut entropy = 0.0;
    for &p in probabilities {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}
