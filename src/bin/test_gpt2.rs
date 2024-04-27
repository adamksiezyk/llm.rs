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
    let mut train_loader = DataLoader::new(train_tokens.to_str().unwrap());
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
    // input handling and its state
    tokens_file: File,
    file_size: u64,
    current_position: u64,
    // convenience variables
    num_batches: u64,
}

impl DataLoader {
    fn new(filename: &str) -> DataLoader {
        let file = File::open(filename).expect(&format!("Could not open file: '{filename}'"));
        let file_size = file.metadata().expect("Culd not read file metadata").len();

        if file_size < ((B * T + 1) * size_of::<u32>()) as u64 {
            panic!("Error: file size is too small for the batch size and sequence length");
        }

        DataLoader {
            tokens_file: file,
            file_size,
            current_position: 0,
            num_batches: file_size / (B * T * size_of::<u32>()) as u64,
        }
    }

    fn next_batch(&mut self) -> [u32; B * T + 1] {
        // if we are at the end of the file, loop back to the beginning
        if self.current_position + ((B * T + 1) * size_of::<u32>()) as u64 > self.file_size {
            self.current_position = 0;
        }

        self.current_position += (B * T * size_of::<u32>()) as u64;
        read_u32(&mut self.tokens_file, B * T + 1)
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

struct GPT2Config {
    max_seq_len: u32, // max sequence length, e.g. 1024
    vocab_size: u32,  // vocab size, e.g. 50257
    num_layers: u32,  // number of layers, e.g. 12
    num_heads: u32,   // number of heads in attention, e.g. 12
    channels: u32,    // number of channels, e.g. 768
}

struct GPT2 {
    config: GPT2Config,
    // the weights (parameters) of the model, and their sizes
    params: ParameterTensors,
    param_sizes: [usize; NUM_PARAMETER_TENSORS],
    num_parameters: usize,
    // gradients of the weights
    grads: ParameterTensors,
    grads_memory: Vec<f32>,
    // buffers for the AdamW optimizer
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
    // the activations of the model, and their sizes
    acts: ActivationTensors,
    act_sizes: [usize; NUM_ACTIVATION_TENSORS],
    acts_memory: Vec<f32>,
    num_activations: usize,
    // gradients of the activations
    grads_acts: ActivationTensors,
    grads_acts_memory: Vec<f32>,
    // other run state configuration
    batch_size: u32,   // the batch size (B) of current forward pass
    seq_len: u32,      // the sequence length (T) of current forward pass
    inputs: Vec<u32>,  // the input tokens for the current forward pass
    targets: Vec<u32>, // the target tokens for the current forward pass
    mean_loss: f32,    // after a forward pass with targets, will be populated with the mean loss
}

impl GPT2 {
    fn new(checkpoint_path: &str) {
        // read in model from a checkpoint file
        let mut model_file =
            File::open(checkpoint_path).expect(&format!("Could not open file '{checkpoint_path}'"));
        let model_header = read_i32(&mut model_file, 256);
        assert!(model_header[0] == 20240326);
        assert!(model_header[1] == 1);

        // read in hyperparameters
        let max_t = model_header[2];
        let v = model_header[3];
        let l = model_header[4];
        let nh = model_header[5];
        let c = model_header[6];
        println!("[GPT-2]");
        println!("max_seq_len: {max_t}");
        println!("vocab_size: {v}");
        println!("num_layers: {l}");
        println!("num_heads: {nh}");
        println!("channels: {c}");

        // allocate space for all the parameters and read them in
        let num_parameters = (v * c
            + max_t * c
            + l * c
            + l * c
            + l * (3 * c) * c
            + l * (3 * c)
            + l * c * c
            + l * c
            + l * c
            + l * c
            + l * (4 * c) * c
            + l * (4 * c)
            + l * c * (4 * c)
            + l * c
            + c
            + c) as usize;
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

        // count the number of parameters
        println!("num_parameters: {num_parameters}");

        // read in all the parameters from file
        let params = ParameterTensors {
            wte: read_f32(&mut model_file, (v * c) as usize),
            wpe: read_f32(&mut model_file, (max_t * c) as usize),
            ln1w: read_f32(&mut model_file, (l * c) as usize),
            ln1b: read_f32(&mut model_file, (l * c) as usize),
            qkvw: read_f32(&mut model_file, (l * (3 * c) * c) as usize),
            qkvb: read_f32(&mut model_file, (l * (3 * c)) as usize),
            attprojw: read_f32(&mut model_file, (l * c * c) as usize),
            attprojb: read_f32(&mut model_file, (l * c) as usize),
            ln2w: read_f32(&mut model_file, (l * c) as usize),
            ln2b: read_f32(&mut model_file, (l * c) as usize),
            fcw: read_f32(&mut model_file, (l * (4 * c) * c) as usize),
            fcb: read_f32(&mut model_file, (l * (4 * c)) as usize),
            fcprojw: read_f32(&mut model_file, (l * c * (4 * c)) as usize),
            fcprojb: read_f32(&mut model_file, (l * c) as usize),
            lnfw: read_f32(&mut model_file, c as usize),
            lnfb: read_f32(&mut model_file, c as usize),
        };
    }
}
