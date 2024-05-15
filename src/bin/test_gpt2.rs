use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    mem::{size_of, take},
    path::Path,
    usize,
};

// hyperparameters
const B: usize = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
const T: usize = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
const NUM_PARAMETER_TENSORS: usize = 16;
const NUM_ACTIVATION_TENSORS: usize = 23;

fn main() {
    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    let tiny_stories_train = Path::new("data/TinyStories_train.bin");
    let tiny_stories_val = Path::new("data/TinyStories_val.bin");
    let tiny_shakespeare_train = Path::new("data/tiny_shakespeare_train.bin");
    let tiny_shakespeare_val = Path::new("data/tiny_shakespeare_val.bin");
    let train_tokens = if tiny_shakespeare_train.exists() {
        tiny_shakespeare_train
    } else {
        tiny_stories_train
    };
    let val_tokens = if tiny_shakespeare_val.exists() {
        tiny_shakespeare_val
    } else {
        tiny_stories_val
    };
    let tokenizer = Tokenizer::new("gpt2_tokenizer.bin");
    let mut train_loader = DataLoader::new(train_tokens.to_str().unwrap(), B, T);
    // let sentence = train_loader
    //     .next_batch()
    //     .iter()
    //     .filter_map(|t| tokenizer.decode(*t).ok())
    //     .map(|t| t.to_string())
    //     .collect::<Vec<_>>()
    //     .join("");
    // println!("{sentence}");

    GPT2::new("gpt2_124M.bin");
}

fn read_u32(file: &mut File, size: usize) -> Vec<u32> {
    let mut buffer = vec![0u8; size * size_of::<u32>()];
    file.read(&mut buffer).unwrap();

    buffer
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>()
}

fn read_i32(file: &mut File, size: usize) -> Vec<i32> {
    let mut buffer = vec![0u8; size * size_of::<i32>()];
    file.read(&mut buffer).unwrap();

    buffer
        .chunks_exact(size_of::<i32>())
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>()
}

fn read_f32(file: &mut File, size: usize) -> Vec<f32> {
    let mut buffer = vec![0u8; size * size_of::<f32>()];
    file.read(&mut buffer).unwrap();

    buffer
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>()
}

struct DataLoader {
    // hyperparameters
    b: usize,
    t: usize,
    // input handling and its state
    tokens_file: File,
    file_size: u64,
    current_position: u64,
    // convenience variables
    num_batches: u64,
}

impl DataLoader {
    fn new(filename: &str, b: usize, t: usize) -> DataLoader {
        let file = File::open(filename).expect(&format!("Could not open file: '{filename}'"));
        let file_size = file.metadata().expect("Culd not read file metadata").len();

        if file_size < ((b * t + 1) * size_of::<u32>()) as u64 {
            panic!("Error: file size is too small for the batch size and sequence length");
        }

        DataLoader {
            b,
            t,
            tokens_file: file,
            file_size,
            current_position: 0,
            num_batches: file_size / (b * t * size_of::<u32>()) as u64,
        }
    }

    fn next_batch(&mut self) -> Vec<u32> {
        // if we are at the end of the file, loop back to the beginning
        if self.current_position + ((self.b * self.t + 1) * size_of::<u32>()) as u64
            > self.file_size
        {
            self.current_position = 0;
        }

        self.current_position += (self.b * self.t * size_of::<u32>()) as u64;
        read_u32(&mut self.tokens_file, self.b * self.t + 1)
            .try_into()
            .expect("Error while reading batch from file")
    }
}

struct Tokenizer {
    vocab_size: u32,
    token_table: HashMap<u32, String>,
}

impl Tokenizer {
    fn new(filename: &str) -> Tokenizer {
        let mut file = File::open(filename).expect(&format!("Could not open file: '{filename}'"));

        // read in the header
        let header = read_u32(&mut file, 256);
        assert!(header[0] == 20240328);
        assert!(header[1] == 1);
        let vocab_size = header[2];
        // read in all the tokens
        let mut length_buff = [0u8; 1];
        let mut token_table = HashMap::<u32, String>::new();
        for i in 0..vocab_size {
            file.read(&mut length_buff).unwrap();
            let mut buff = vec![0u8; length_buff[0] as usize];
            file.read(&mut buff).unwrap();
            let buff_u16 = buff.iter().map(|t| *t as u16).collect::<Vec<_>>();
            let token = String::from_utf16(&buff_u16).unwrap();
            token_table.insert(i, token);
        }

        Tokenizer {
            vocab_size,
            token_table,
        }
    }

    fn decode(&self, token_id: u32) -> Result<&str, &'static str> {
        if token_id < self.vocab_size {
            Ok(&self.token_table[&token_id])
        } else {
            Err("Invalid token")
        }
    }
}

struct ParameterTensors {
    wte: Vec<f32>,      // (V, C)
    wpe: Vec<f32>,      // (maxT, C)
    ln1w: Vec<f32>,     // (L, C)
    ln1b: Vec<f32>,     // (L, C)
    qkvw: Vec<f32>,     // (L, 3*C, C)
    qkvb: Vec<f32>,     // (L, 3*C)
    attprojw: Vec<f32>, // (L, C, C)
    attprojb: Vec<f32>, // (L, C)
    ln2w: Vec<f32>,     // (L, C)
    ln2b: Vec<f32>,     // (L, C)
    fcw: Vec<f32>,      // (L, 4*C, C)
    fcb: Vec<f32>,      // (L, 4*C)
    fcprojw: Vec<f32>,  // (L, C, 4*C)
    fcprojb: Vec<f32>,  // (L, C)
    lnfw: Vec<f32>,     // (C)
    lnfb: Vec<f32>,     // (C)
}

impl ParameterTensors {
    fn new(v: usize, c: usize, l: usize, max_t: usize) -> ParameterTensors {
        ParameterTensors {
            wte: vec![0f32; v * c],
            wpe: vec![0f32; max_t * c],
            ln1w: vec![0f32; l * c],
            ln1b: vec![0f32; l * c],
            qkvw: vec![0f32; l * 3 * c * c],
            qkvb: vec![0f32; l * 3 * c],
            attprojw: vec![0f32; l * c * c],
            attprojb: vec![0f32; l * c],
            ln2w: vec![0f32; l * c],
            ln2b: vec![0f32; l * c],
            fcw: vec![0f32; l * 4 * c * c],
            fcb: vec![0f32; l * 4 * c],
            fcprojw: vec![0f32; l * c * 4 * c],
            fcprojb: vec![0f32; l * c],
            lnfw: vec![0f32; c],
            lnfb: vec![0f32; c],
        }
    }
}

struct ActivationTensors {
    encoded: Vec<f32>,   // (B, T, C)
    ln1: Vec<f32>,       // (L, B, T, C)
    ln1_mean: Vec<f32>,  // (L, B, T)
    ln1_rstd: Vec<f32>,  // (L, B, T)
    qkv: Vec<f32>,       // (L, B, T, 3*C)
    atty: Vec<f32>,      // (L, B, T, C)
    preatt: Vec<f32>,    // (L, B, NH, T, T)
    att: Vec<f32>,       // (L, B, NH, T, T)
    attproj: Vec<f32>,   // (L, B, T, C)
    residual2: Vec<f32>, // (L, B, T, C)
    ln2: Vec<f32>,       // (L, B, T, C)
    ln2_mean: Vec<f32>,  // (L, B, T)
    ln2_rstd: Vec<f32>,  // (L, B, T)
    fch: Vec<f32>,       // (L, B, T, 4*C)
    fch_gelu: Vec<f32>,  // (L, B, T, 4*C)
    fcproj: Vec<f32>,    // (L, B, T, C)
    residual3: Vec<f32>, // (L, B, T, C)
    lnf: Vec<f32>,       // (B, T, C)
    lnf_mean: Vec<f32>,  // (B, T)
    lnf_rstd: Vec<f32>,  // (B, T)
    logits: Vec<f32>,    // (B, T, V)
    probs: Vec<f32>,     // (B, T, V)
    losses: Vec<f32>,    // (B, T)
}

impl ActivationTensors {
    fn new(b: usize, t: usize, l: usize, c: usize, v: usize, nh: usize) -> ActivationTensors {
        ActivationTensors {
            encoded: vec![0f32; b * t * c],
            ln1: vec![0f32; l * b * t * c],
            ln1_mean: vec![0f32; l * b * t],
            ln1_rstd: vec![0f32; l * b * t],
            qkv: vec![0f32; l * b * t * 3 * c],
            atty: vec![0f32; l * b * t * c],
            preatt: vec![0f32; l * b * nh * t * t],
            att: vec![0f32; l * b * nh * t * t],
            attproj: vec![0f32; l * b * t * c],
            residual2: vec![0f32; l * b * t * c],
            ln2: vec![0f32; l * b * t * c],
            ln2_mean: vec![0f32; l * b * t],
            ln2_rstd: vec![0f32; l * b * t],
            fch: vec![0f32; l * b * t * 4 * c],
            fch_gelu: vec![0f32; l * b * t * 4 * c],
            fcproj: vec![0f32; l * b * t * c],
            residual3: vec![0f32; l * b * t * c],
            lnf: vec![0f32; b * t * c],
            lnf_mean: vec![0f32; b * t],
            lnf_rstd: vec![0f32; b * t],
            logits: vec![0f32; b * t * v],
            probs: vec![0f32; b * t * v],
            losses: vec![0f32; b * t],
        }
    }
}

struct GPT2Config {
    max_seq_len: usize, // max sequence length, e.g. 1024
    vocab_size: usize,  // vocab size, e.g. 50257
    num_layers: usize,  // number of layers, e.g. 12
    num_heads: usize,   // number of heads in attention, e.g. 12
    channels: usize,    // number of channels, e.g. 768
}

struct GPT2 {
    config: GPT2Config,
    // the weights (parameters) of the model, and their sizes
    params: ParameterTensors,
    param_sizes: [usize; NUM_PARAMETER_TENSORS],
    num_parameters: usize,
    // gradients of the weights
    grads: ParameterTensors,
    // buffers for the AdamW optimizer
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
    // the activations of the model, and their sizes
    acts: ActivationTensors,
    // gradients of the activations
    grads_acts: ActivationTensors,
    // other run state configuration
    batch_size: u32,   // the batch size (B) of current forward pass
    seq_len: u32,      // the sequence length (T) of current forward pass
    inputs: Vec<u32>,  // the input tokens for the current forward pass
    targets: Vec<u32>, // the target tokens for the current forward pass
    mean_loss: f32,    // after a forward pass with targets, will be populated with the mean loss
}

impl GPT2 {
    fn new(checkpoint_path: &str) -> GPT2 {
        // read in model from a checkpoint file
        let mut model_file =
            File::open(checkpoint_path).expect(&format!("Could not open file '{checkpoint_path}'"));
        let model_header = read_i32(&mut model_file, 256);
        assert!(model_header[0] == 20240326);
        assert!(model_header[1] == 1);

        // read in hyperparameters
        let max_t = model_header[2] as usize;
        let v = model_header[3] as usize;
        let l = model_header[4] as usize;
        let nh = model_header[5] as usize;
        let c = model_header[6] as usize;
        let config = GPT2Config {
            max_seq_len: max_t,
            vocab_size: v,
            num_layers: l,
            num_heads: nh,
            channels: c,
        };
        println!("[GPT-2]");
        println!("max_seq_len: {max_t}");
        println!("vocab_size: {v}");
        println!("num_layers: {l}");
        println!("num_heads: {nh}");
        println!("channels: {c}");

        // allocate space for parameters and read them in
        let param_sizes = [
            (v * c) as usize,           // wte
            (max_t * c) as usize,       // wpe
            (l * c) as usize,           // ln1w
            (l * c) as usize,           // ln1b
            (l * (3 * c) * c) as usize, // qkvw
            (l * (3 * c)) as usize,     // qkvb
            (l * c * c) as usize,       // attprojw
            (l * c) as usize,           // attprojb
            (l * c) as usize,           // ln2w
            (l * c) as usize,           // ln2b
            (l * (4 * c) * c) as usize, // fcw
            (l * (4 * c)) as usize,     // fcb
            (l * c * (4 * c)) as usize, // fcprojw
            (l * c) as usize,           // fcprojb
            c as usize,                 // lnfw
            c as usize,                 // lnfb
        ];
        let num_parameters = param_sizes.iter().sum();

        // count the number of parameters
        println!("num_parameters: {num_parameters}");

        // read in parameters from file
        let params = ParameterTensors {
            wte: read_f32(&mut model_file, param_sizes[0]),
            wpe: read_f32(&mut model_file, param_sizes[1]),
            ln1w: read_f32(&mut model_file, param_sizes[2]),
            ln1b: read_f32(&mut model_file, param_sizes[3]),
            qkvw: read_f32(&mut model_file, param_sizes[4]),
            qkvb: read_f32(&mut model_file, param_sizes[5]),
            attprojw: read_f32(&mut model_file, param_sizes[6]),
            attprojb: read_f32(&mut model_file, param_sizes[7]),
            ln2w: read_f32(&mut model_file, param_sizes[8]),
            ln2b: read_f32(&mut model_file, param_sizes[9]),
            fcw: read_f32(&mut model_file, param_sizes[10]),
            fcb: read_f32(&mut model_file, param_sizes[11]),
            fcprojw: read_f32(&mut model_file, param_sizes[12]),
            fcprojb: read_f32(&mut model_file, param_sizes[13]),
            lnfw: read_f32(&mut model_file, param_sizes[14]),
            lnfb: read_f32(&mut model_file, param_sizes[15]),
        };

        GPT2 {
            config,
            params,
            param_sizes,
            num_parameters,
            grads: ParameterTensors::new(0usize, 0usize, 0usize, 0usize),
            // buffers for the AdamW optimizer
            m_memory: vec![],
            v_memory: vec![],
            // the activations of the model, and their sizes
            acts: ActivationTensors::new(B, T, l, c, v, nh),
            grads_acts: ActivationTensors::new(B, T, l, c, v, nh),
            seq_len: 0,
            batch_size: 0,
            inputs: vec![],
            targets: vec![],
            mean_loss: -1.0,
        }
    }

    fn forward(&self, inputs: Vec<f32>, targets: Option<Vec<f32>>) {
        // validate inputs, all indices must be in the range [0, V)
        for i in 0usize..(B * T) {
            assert!(0f32 <= inputs[i] && inputs[i] < self.config.vocab_size as f32);
            match &targets {
                Some(t) => assert!(0f32 <= t[i] && t[i] < self.config.vocab_size as f32),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader_first_batch() {
        let mut data_loader = DataLoader::new("data/tiny_shakespeare_train.bin", 1, 32);
        let batch = data_loader.next_batch();
        let expected = [
            9203, 262, 39418, 13, 628, 50256, 22747, 49208, 805, 25, 198, 9203, 262, 39418, 0, 628,
            50256, 44879, 40, 3535, 1565, 2937, 25, 198, 42012, 13, 628, 50256, 22747, 49208, 805,
            25, 198,
        ];
        assert_eq!(batch, expected);
    }

    #[test]
    fn test_data_loader_second_batch() {
        let mut data_loader = DataLoader::new("data/tiny_shakespeare_train.bin", 1, 32);
        let _ = data_loader.next_batch();
        let batch = data_loader.next_batch();
        let expected = [
            8496, 338, 326, 30, 628, 50256, 44879, 40, 3535, 1565, 2937, 25, 198, 40, 6, 262, 1748,
            286, 479, 2737, 290, 269, 8516, 13, 628, 50256, 22747, 49208, 805, 25, 198, 40, 6,
        ];
        assert_eq!(batch, expected);
    }

    #[test]
    fn test_tokenizer_decode() {
        let tokenizer = Tokenizer::new("gpt2_tokenizer.bin");
        let inputs = [
            9203u32, 262, 39418, 13, 628, 50256, 22747, 49208, 805, 25, 198, 9203, 262, 39418, 0,
            628, 50256, 44879, 40, 3535, 1565, 2937, 25, 198, 42012, 13, 628, 50256, 22747, 49208,
            805, 25, 198, 8496, 338, 326, 30, 628, 50256,
        ];
        let tokens = inputs
            .iter()
            .filter_map(|t| tokenizer.decode(*t).ok())
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("");
        let expected = "Under the canopy.

<|endoftext|>Third Servingman:
Under the canopy!

<|endoftext|>CORIOLANUS:
Ay.

<|endoftext|>Third Servingman:
Where's that?

<|endoftext|>";
        assert_eq!(tokens, expected);
    }
}
