// Audio Test Data Generator for Morse Code AI Training
// Generates labeled audio samples with precise morse code patterns

use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::io::Write;

use serde::{Serialize, Deserialize};

use rand::Rng;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MorseAudioSample {
    pub audio_data: Vec<f32>,
    pub labels: Vec<MorseLabel>,
    pub metadata: SampleMetadata,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MorseLabel {
    pub symbol: MorseSymbol,
    pub start_sample: usize,
    pub end_sample: usize,
    pub confidence: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MorseSymbol {
    Dot,
    Dash,
    IntraCharGap,
    CharGap,
    WordGap,
    Noise,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SampleMetadata {
    pub sample_rate: u32,
    pub frequency: f32,
    pub wpm: u32,
    pub noise_level: f32,
    pub snr_db: f32,
    pub text: String,
    pub duration_ms: u32,
}

pub struct MorseDataGenerator {
    sample_rate: u32,
    base_frequency: f32,
    noise_generators: Vec<NoiseGenerator>,
}

pub struct NoiseGenerator {
    noise_type: NoiseType,
    amplitude: f32,
}

#[derive(Debug, Clone)]
pub enum NoiseType {
    WhiteNoise,
    PinkNoise,
    BrownNoise,
    RadioStatic,
    HarmoniDistortion,
}

impl MorseDataGenerator {
    pub fn new(sample_rate: u32, base_frequency: f32) -> Self {
        let noise_generators = vec![
            NoiseGenerator { noise_type: NoiseType::WhiteNoise, amplitude: 0.05 },
            NoiseGenerator { noise_type: NoiseType::RadioStatic, amplitude: 0.03 },
            NoiseGenerator { noise_type: NoiseType::HarmoniDistortion, amplitude: 0.02 },
        ];

        Self {
            sample_rate,
            base_frequency,
            noise_generators,
        }
    }

    pub fn generate_training_dataset(&self, data_dir: &str, num_samples: usize) -> Result<(), Box<dyn std::error::Error>> {
        create_dir_all(data_dir)?;
        
        // Create diverse training texts
        let training_texts = self.generate_training_texts(num_samples);
        
        println!("Generating {} training samples...", num_samples);
        
        let mut rng = rand::thread_rng();
        let mut min_snr = f32::MAX;
        let mut max_snr = f32::MIN;
        for (i, text) in training_texts.iter().enumerate() {
            let wpm = 15 + (i % 20) as u32; // WPM from 15-35
            // Randomize frequency between 400 and 1200 Hz
            let frequency = 400.0 + rng.gen::<f32>() * 800.0;
            // Randomly select target SNR: either -22 or -7 dB (exclusive)
                let target_snr = if rng.gen_bool(0.5) { -23.0 } else { -7.0 };
            // Start with a reasonable noise level
            let mut noise_level = 0.25;
            let mut morse_amplitude = 0.07;
            let mut actual_snr = 0.0;
            // Iteratively adjust amplitude to approach target SNR
            for _ in 0..10 {
                let sample = self.generate_morse_sample_with_amplitude(text, wpm, frequency, noise_level, morse_amplitude)?;
                actual_snr = sample.metadata.snr_db;
                if actual_snr < target_snr {
                    morse_amplitude *= 1.1;
                } else {
                    morse_amplitude *= 0.9;
                }
            }
            // Final sample with tuned amplitude
            let sample = self.generate_morse_sample_with_amplitude(text, wpm, frequency, noise_level, morse_amplitude)?;
            if sample.metadata.snr_db < min_snr { min_snr = sample.metadata.snr_db; }
            if sample.metadata.snr_db > max_snr { max_snr = sample.metadata.snr_db; }
            let sample_file = format!("{}/sample_{:06}.bin", data_dir, i);
            let label_file = format!("{}/sample_{:06}.json", data_dir, i);
            self.save_sample_binary(&sample, &sample_file)?;
            self.save_labels_json(&sample, &label_file)?;
        }
        println!("SNR range for this batch: {:.2} dB to {:.2} dB", min_snr, max_snr);
        
        println!("Dataset generation complete!");
        Ok(())
    }

    fn generate_training_texts(&self, num_samples: usize) -> Vec<String> {
        let mut texts = Vec::new();
        
        // Single characters
        for ch in 'A'..='Z' {
            texts.push(ch.to_string());
        }
        for ch in '0'..='9' {
            texts.push(ch.to_string());
        }
        
        // Common words
        let common_words = vec![
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE",
            "OUR", "HAD", "DAY", "GET", "USE", "MAN", "NEW", "NOW", "OLD", "SEE", "HIM", "TWO",
            "HOW", "ITS", "WHO", "OIL", "SIT", "SET", "RUN", "EAT", "FAR", "SEA", "EYE", "BAD",
        ];
        
        for word in &common_words {
            texts.push(word.to_string());
        }
        
        // Short phrases
        let phrases = vec![
            "SOS", "CQ CQ", "QRT", "QRV", "73", "DE W1AW", "TEST", "QSO", "QTH", "QSL",
            "HELLO WORLD", "MORSE CODE", "RADIO TEST", "AI TRAINING", "DEEP LEARNING",
        ];
        
        for phrase in &phrases {
            texts.push(phrase.to_string());
        }
        
        // Generate random combinations to reach num_samples
        let mut rng = rand::thread_rng();
        while texts.len() < num_samples {
            let len = rng.gen_range(2..=8);
            let mut random_text = String::new();
            
            for i in 0..len {
                if i > 0 { random_text.push(' '); }
                if rng.gen_bool(0.7) {
                    // Letter
                    random_text.push(rng.gen_range(b'A'..=b'Z') as char);
                } else {
                    // Number
                    random_text.push(rng.gen_range(b'0'..=b'9') as char);
                }
            }
            texts.push(random_text);
        }
        
        texts.truncate(num_samples);
        texts
    }

    pub fn generate_morse_sample(&self, text: &str, wpm: u32, frequency: f32, noise_level: f32) -> Result<MorseAudioSample, Box<dyn std::error::Error>> {
        // Default amplitude for backward compatibility (used in SNR tuning loop)
        self.generate_morse_sample_with_amplitude(text, wpm, frequency, noise_level, 0.07)
    }

    pub fn generate_morse_sample_with_amplitude(&self, text: &str, wpm: u32, frequency: f32, noise_level: f32, morse_amplitude: f32) -> Result<MorseAudioSample, Box<dyn std::error::Error>> {
        let morse_map = create_morse_map();
        let reverse_map: HashMap<char, String> = morse_map.into_iter().map(|(k, v)| (v, k)).collect();
        let mut audio_data = Vec::new();
        let mut labels = Vec::new();
        let mut clean_signal = Vec::new();
        
        // Timing calculations (standard morse timing)
        let dot_duration_ms = 1200.0 / wpm as f32;
        let dash_duration_ms = dot_duration_ms * 3.0;
        let intra_char_gap_ms = dot_duration_ms;
        let char_gap_ms = dot_duration_ms * 3.0;
        let word_gap_ms = dot_duration_ms * 7.0;
        
        let dot_samples = (dot_duration_ms * self.sample_rate as f32 / 1000.0) as usize;
        let dash_samples = (dash_duration_ms * self.sample_rate as f32 / 1000.0) as usize;
        let intra_gap_samples = (intra_char_gap_ms * self.sample_rate as f32 / 1000.0) as usize;
        let char_gap_samples = (char_gap_ms * self.sample_rate as f32 / 1000.0) as usize;
        let word_gap_samples = (word_gap_ms * self.sample_rate as f32 / 1000.0) as usize;
        
        for (word_idx, word) in text.split_whitespace().enumerate() {
            if word_idx > 0 {
                // Word gap
                let start_sample = audio_data.len();
                let silence = self.generate_silence(word_gap_samples);
                audio_data.extend(silence.iter());
                clean_signal.extend(silence.iter());
                labels.push(MorseLabel {
                    symbol: MorseSymbol::WordGap,
                    start_sample,
                    end_sample: audio_data.len(),
                    confidence: 1.0,
                });
            }
            
            for (char_idx, ch) in word.chars().enumerate() {
                if char_idx > 0 {
                    // Character gap
                    let start_sample = audio_data.len();
                    let silence = self.generate_silence(char_gap_samples);
                    audio_data.extend(silence.iter());
                    clean_signal.extend(silence.iter());
                    labels.push(MorseLabel {
                        symbol: MorseSymbol::CharGap,
                        start_sample,
                        end_sample: audio_data.len(),
                        confidence: 1.0,
                    });
                }
                
                if let Some(pattern) = reverse_map.get(&ch.to_ascii_uppercase()) {
                    for (symbol_idx, morse_char) in pattern.chars().enumerate() {
                        if symbol_idx > 0 {
                            // Intra-character gap
                            let start_sample = audio_data.len();
                            let silence = self.generate_silence(intra_gap_samples);
                            audio_data.extend(silence.iter());
                            clean_signal.extend(silence.iter());
                            labels.push(MorseLabel {
                                symbol: MorseSymbol::IntraCharGap,
                                start_sample,
                                end_sample: audio_data.len(),
                                confidence: 1.0,
                            });
                        }
                        
                        match morse_char {
                            '.' => {
                                let start_sample = audio_data.len();
                                let tone = self.generate_tone_with_amplitude(dot_samples, frequency, morse_amplitude);
                                audio_data.extend(tone.iter());
                                clean_signal.extend(tone.iter());
                                labels.push(MorseLabel {
                                    symbol: MorseSymbol::Dot,
                                    start_sample,
                                    end_sample: audio_data.len(),
                                    confidence: 1.0,
                                });
                            },
                            '-' => {
                                let start_sample = audio_data.len();
                                let tone = self.generate_tone_with_amplitude(dash_samples, frequency, morse_amplitude);
                                audio_data.extend(tone.iter());
                                clean_signal.extend(tone.iter());
                                labels.push(MorseLabel {
                                    symbol: MorseSymbol::Dash,
                                    start_sample,
                                    end_sample: audio_data.len(),
                                    confidence: 1.0,
                                });
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
        
        // Save clean signal for SNR calculation
        let mut noisy_signal = audio_data.clone();
        self.add_noise(&mut noisy_signal, noise_level);
        
        // Calculate SNR (in dB): SNR = 10*log10(signal_power/noise_power)
        let signal_power: f32 = clean_signal.iter().map(|x| x * x).sum::<f32>() / clean_signal.len().max(1) as f32;
    let noise_power: f32 = noisy_signal.iter().zip(clean_signal.iter()).map(|(n, s): (&f32, &f32)| (n - s).powi(2)).sum::<f32>() / clean_signal.len().max(1) as f32;
        let snr_db = if noise_power > 0.0 && signal_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            0.0
        };
        
        let metadata = SampleMetadata {
            sample_rate: self.sample_rate,
            frequency,
            wpm,
            noise_level,
            snr_db,
            text: text.to_string(),
            duration_ms: (noisy_signal.len() as f32 / self.sample_rate as f32 * 1000.0) as u32,
        };
        
        Ok(MorseAudioSample {
            audio_data: noisy_signal,
            labels,
            metadata,
        })
    }

    fn generate_tone_with_amplitude(&self, num_samples: usize, frequency: f32, morse_amplitude: f32) -> Vec<f32> {
        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / self.sample_rate as f32;
            let progress = i as f32 / num_samples as f32;
            let envelope = if progress < 0.1 {
                progress * 10.0
            } else if progress > 0.9 {
                (1.0 - progress) * 10.0
            } else {
                1.0
            };
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * morse_amplitude * envelope;
            samples.push(sample);
        }
        samples
    }

    fn generate_silence(&self, num_samples: usize) -> Vec<f32> {
        vec![0.0; num_samples]
    }

    fn add_noise(&self, audio_data: &mut [f32], noise_level: f32) {
        let mut rng = rand::thread_rng();
        
        for sample in audio_data.iter_mut() {
            // White noise
            let white_noise = (rng.gen::<f32>() - 0.5) * noise_level;
            *sample += white_noise;
            
            // Clamp to prevent clipping
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    fn save_sample_binary(&self, sample: &MorseAudioSample, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(filename)?;
        
        // Write audio data as binary f32
        for &value in &sample.audio_data {
            file.write_all(&value.to_le_bytes())?;
        }
        
        Ok(())
    }

    fn save_labels_json(&self, sample: &MorseAudioSample, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Only serialize labels and metadata, not audio_data
        let json_data = serde_json::json!({
            "labels": sample.labels,
            "metadata": sample.metadata,
        });
        std::fs::write(filename, serde_json::to_string_pretty(&json_data)?)?;
        Ok(())
    }

    pub fn generate_test_data(&self, data_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        create_dir_all(data_dir)?;

        let text = "TEST RADIO MORSE";
        let wpm = 20;
        let frequency = self.base_frequency + 100.0; // Slightly higher frequency
        let noise_level = 0.05; // Moderate noise

        let mut sample = self.generate_morse_sample(text, wpm, frequency, noise_level)?;

        // Add fading effect
        self.add_fading(&mut sample.audio_data);

        // Update metadata
        sample.metadata.noise_level = noise_level;
        sample.metadata.text = format!("{} (with noise and fading)", text);

        // Save as binary and JSON
        let sample_file = format!("{}/test_sample.bin", data_dir);
        let label_file = format!("{}/test_sample.json", data_dir);

        self.save_sample_binary(&sample, &sample_file)?;
        let json_data = sample.to_metadata_json();
        std::fs::write(label_file, serde_json::to_string_pretty(&json_data)?)?;

        println!("Test data generation complete!");
        Ok(())
    }

    fn add_fading(&self, audio_data: &mut [f32]) {
        let len = audio_data.len();
        for (i, sample) in audio_data.iter_mut().enumerate() {
            let progress = i as f32 / len as f32;
            let fade_factor = if progress < 0.1 {
                progress * 10.0
            } else if progress > 0.9 {
                (1.0 - progress) * 10.0
            } else {
                1.0
            };
            *sample *= fade_factor;
        }
    }
}

impl MorseAudioSample {
    pub fn to_metadata_json(&self) -> serde_json::Value {
        serde_json::json!({
            "labels": self.labels,
            "metadata": self.metadata,
        })
    }
}

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
