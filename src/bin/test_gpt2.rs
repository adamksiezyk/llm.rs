use std::{collections::HashMap, fs::File, io::Read, mem::size_of, path::Path, usize};

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
    let sentence = train_loader
        .next_batch()
        .iter()
        .filter_map(|t| tokenizer.decode(*t).ok())
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join("");
    println!("{sentence}\n");

    let gpt2 = GPT2::from_checkpoint("gpt2_124M.bin", B, T);
    println!("[GPT-2]");
    println!("max_seq_len: {}", gpt2.config.max_seq_len);
    println!("vocab_size: {}", gpt2.config.vocab_size);
    println!("num_layers: {}", gpt2.config.num_layers);
    println!("num_heads: {}", gpt2.config.num_heads);
    println!("channels: {}", gpt2.config.channels);
    println!("num_parameters: {}", gpt2.num_parameters);
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

struct ParameterTensors<'a> {
    wte: &'a [f32],      // (V, C)
    wpe: &'a [f32],      // (maxT, C)
    ln1w: &'a [f32],     // (L, C)
    ln1b: &'a [f32],     // (L, C)
    qkvw: &'a [f32],     // (L, 3*C, C)
    qkvb: &'a [f32],     // (L, 3*C)
    attprojw: &'a [f32], // (L, C, C)
    attprojb: &'a [f32], // (L, C)
    ln2w: &'a [f32],     // (L, C)
    ln2b: &'a [f32],     // (L, C)
    fcw: &'a [f32],      // (L, 4*C, C)
    fcb: &'a [f32],      // (L, 4*C)
    fcprojw: &'a [f32],  // (L, C, 4*C)
    fcprojb: &'a [f32],  // (L, C)
    lnfw: &'a [f32],     // (C)
    lnfb: &'a [f32],     // (C)
}

impl<'a> ParameterTensors<'a> {
    fn new<'b>(
        paramsmemory: &'b [f32],
        param_sizes: &[usize; NUM_PARAMETER_TENSORS],
    ) -> ParameterTensors<'b> {
        let mut params = Vec::with_capacity(NUM_PARAMETER_TENSORS);
        let mut ptr = 0usize;
        for s in param_sizes.iter() {
            params.push(&paramsmemory[ptr..ptr + s]);
            ptr += s;
        }
        ParameterTensors {
            wte: params[0],
            wpe: params[1],
            ln1w: params[2],
            ln1b: params[3],
            qkvw: params[4],
            qkvb: params[5],
            attprojw: params[6],
            attprojb: params[7],
            ln2w: params[8],
            ln2b: params[9],
            fcw: params[10],
            fcb: params[11],
            fcprojw: params[12],
            fcprojb: params[13],
            lnfw: params[14],
            lnfb: params[15],
        }
    }
}

struct ActivationTensors<'a> {
    encoded: &'a [f32],   // (B, T, C)
    ln1: &'a [f32],       // (L, B, T, C)
    ln1_mean: &'a [f32],  // (L, B, T)
    ln1_rstd: &'a [f32],  // (L, B, T)
    qkv: &'a [f32],       // (L, B, T, 3*C)
    atty: &'a [f32],      // (L, B, T, C)
    preatt: &'a [f32],    // (L, B, NH, T, T)
    att: &'a [f32],       // (L, B, NH, T, T)
    attproj: &'a [f32],   // (L, B, T, C)
    residual2: &'a [f32], // (L, B, T, C)
    ln2: &'a [f32],       // (L, B, T, C)
    ln2_mean: &'a [f32],  // (L, B, T)
    ln2_rstd: &'a [f32],  // (L, B, T)
    fch: &'a [f32],       // (L, B, T, 4*C)
    fch_gelu: &'a [f32],  // (L, B, T, 4*C)
    fcproj: &'a [f32],    // (L, B, T, C)
    residual3: &'a [f32], // (L, B, T, C)
    lnf: &'a [f32],       // (B, T, C)
    lnf_mean: &'a [f32],  // (B, T)
    lnf_rstd: &'a [f32],  // (B, T)
    logits: &'a [f32],    // (B, T, V)
    probs: &'a [f32],     // (B, T, V)
    losses: &'a [f32],    // (B, T)
}

impl<'a> ActivationTensors<'a> {
    fn new<'b>(
        acts_memory: &'b [f32],
        act_sizes: &[usize; NUM_ACTIVATION_TENSORS],
    ) -> ActivationTensors<'b> {
        let mut acts = Vec::with_capacity(NUM_ACTIVATION_TENSORS); // : [&[f32]; NUM_PARAMETER_TENSORS];
        let mut ptr = 0usize;
        for s in act_sizes.iter() {
            acts.push(&acts_memory[ptr..ptr + s]);
            ptr += s;
        }
        ActivationTensors {
            encoded: acts[0],
            ln1: acts[1],
            ln1_mean: acts[2],
            ln1_rstd: acts[3],
            qkv: acts[4],
            atty: acts[5],
            preatt: acts[6],
            att: acts[7],
            attproj: acts[8],
            residual2: acts[9],
            ln2: acts[10],
            ln2_mean: acts[11],
            ln2_rstd: acts[12],
            fch: acts[13],
            fch_gelu: acts[14],
            fcproj: acts[15],
            residual3: acts[16],
            lnf: acts[17],
            lnf_mean: acts[18],
            lnf_rstd: acts[19],
            logits: acts[20],
            probs: acts[21],
            losses: acts[22],
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

struct GPT2<'a> {
    config: GPT2Config,
    // the weights (parameters) of the model, and their sizes
    params: ParameterTensors<'a>,
    param_sizes: [usize; NUM_PARAMETER_TENSORS],
    params_memory: Vec<f32>,
    num_parameters: usize,
    // gradients of the weights
    grads: ParameterTensors<'a>,
    grads_memory: Vec<f32>,
    // buffers for the AdamW optimizer
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
    // the activations of the model, and their sizes
    acts: ActivationTensors<'a>,
    // gradients of the activations
    grads_acts: ActivationTensors<'a>,
    // other run state configuration
    batch_size: u32,   // the batch size (B) of current forward pass
    seq_len: u32,      // the sequence length (T) of current forward pass
    inputs: Vec<u32>,  // the input tokens for the current forward pass
    targets: Vec<u32>, // the target tokens for the current forward pass
    mean_loss: f32,    // after a forward pass with targets, will be populated with the mean loss
}

impl<'a> GPT2<'a> {
    fn from_checkpoint(checkpoint_path: &str, b: usize, t: usize) -> GPT2 {
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

        // read in parameters from file
        let params_memory = read_f32(&mut model_file, num_parameters);
        let params_memory_slice =
            unsafe { std::slice::from_raw_parts(params_memory.as_ptr(), params_memory.len()) };
        let params = ParameterTensors::new(params_memory_slice, &param_sizes);

        // zero gradient
        let grads_memory = vec![0f32; num_parameters];
        let grads_memory_slice =
            unsafe { std::slice::from_raw_parts(grads_memory.as_ptr(), grads_memory.len()) };
        let grads = ParameterTensors::new(&grads_memory_slice, &param_sizes);

        // allocate space for activations
        let act_sizes = [
            b * t * c,
            l * b * t * c,
            l * b * t,
            l * b * t,
            l * b * t * 3 * c,
            l * b * t * c,
            l * b * nh * t * t,
            l * b * nh * t * t,
            l * b * t * c,
            l * b * t * c,
            l * b * t * c,
            l * b * t,
            l * b * t,
            l * b * t * 4 * c,
            l * b * t * 4 * c,
            l * b * t * c,
            l * b * t * c,
            b * t * c,
            b * t,
            b * t,
            b * t * v,
            b * t * v,
            b * t,
        ];
        let num_activations: usize = act_sizes.iter().sum();

        // activations
        let act_memory = vec![0f32; num_activations];
        let act_memory_slice =
            unsafe { std::slice::from_raw_parts(act_memory.as_ptr(), act_memory.len()) };
        let acts = ActivationTensors::new(&act_memory_slice, &act_sizes);

        // activations gradients
        let grads_acts_memory = vec![0f32; num_activations];
        let grads_acts_memory_slice = unsafe {
            std::slice::from_raw_parts(grads_acts_memory.as_ptr(), grads_acts_memory.len())
        };
        let grads_acts = ActivationTensors::new(&grads_acts_memory_slice, &act_sizes);

        GPT2 {
            config,
            params,
            param_sizes,
            params_memory,
            num_parameters,
            grads,
            grads_memory,
            //// buffers for the AdamW optimizer
            m_memory: vec![],
            v_memory: vec![],
            //// the activations of the model, and their sizes
            acts,
            grads_acts,
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
