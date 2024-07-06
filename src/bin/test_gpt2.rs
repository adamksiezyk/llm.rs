use std::{collections::HashMap, fs::File, io::Read, path::Path};

// hyperparameters
const B: usize = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
const T: usize = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
const NUM_PARAMETER_TENSORS: usize = 16;
const NUM_ACTIVATION_TENSORS: usize = 23;
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // f32::sqrt(2.0 / std::f32::consts::PI);

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
    let input_tokens = train_loader.next_batch();
    let input_str = input_tokens
        .iter()
        .filter_map(|t| tokenizer.decode(*t).ok())
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join("");
    println!("{input_str}\n");

    let mut gpt2 = GPT2::from_checkpoint("gpt2_124M.bin", B, T);
    println!("[GPT-2]");
    println!("max_seq_len: {}", gpt2.config.max_seq_len);
    println!("vocab_size: {}", gpt2.config.vocab_size);
    println!("num_layers: {}", gpt2.config.num_layers);
    println!("num_heads: {}", gpt2.config.num_heads);
    println!("channels: {}", gpt2.config.channels);
    println!("num_parameters: {}", gpt2.num_parameters);

    let mut input_tokens_2 = input_tokens.clone();
    let mut gen_tokens = Vec::new();
    for _ in 1..T {
        println!("Generating token {} of {}", gen_tokens.len() + 1, T);
        gpt2.forward(&input_tokens_2, B, T);
        let probs = &gpt2.acts.probs;
        let coin = rand::random::<f32>();
        let next_token = sample_mult(probs, gpt2.config.vocab_size, coin);
        input_tokens_2.push(next_token as u32);
        gen_tokens.push(next_token as u32);
        let gen_str = &gen_tokens
            .iter()
            .filter_map(|t| tokenizer.decode(*t).ok())
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("");
        println!("{}", gen_str);
    }
}

fn read_u32(file: &mut File, size: usize) -> Vec<u32> {
    let mut buffer = vec![0u8; size * std::mem::size_of::<u32>()];
    file.read(&mut buffer).unwrap();

    buffer
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>()
}

fn read_i32(file: &mut File, size: usize) -> Vec<i32> {
    let mut buffer = vec![0u8; size * std::mem::size_of::<i32>()];
    file.read(&mut buffer).unwrap();

    buffer
        .chunks_exact(std::mem::size_of::<i32>())
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>()
}

fn read_f32(file: &mut File, size: usize) -> Vec<f32> {
    let mut buffer = vec![0u8; size * std::mem::size_of::<f32>()];
    file.read(&mut buffer).unwrap();

    buffer
        .chunks_exact(std::mem::size_of::<f32>())
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

        if file_size < ((b * t + 1) * std::mem::size_of::<u32>()) as u64 {
            panic!("Error: file size is too small for the batch size and sequence length");
        }

        DataLoader {
            b,
            t,
            tokens_file: file,
            file_size,
            current_position: 0,
            num_batches: file_size / (b * t * std::mem::size_of::<u32>()) as u64,
        }
    }

    fn next_batch(&mut self) -> Vec<u32> {
        // if we are at the end of the file, loop back to the beginning
        if self.current_position + ((self.b * self.t + 1) * std::mem::size_of::<u32>()) as u64
            > self.file_size
        {
            self.current_position = 0;
        }

        self.current_position += (self.b * self.t * std::mem::size_of::<u32>()) as u64;
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
    encoded: &'a mut [f32],   // (B, T, C)
    ln1: &'a mut [f32],       // (L, B, T, C)
    ln1_mean: &'a mut [f32],  // (L, B, T)
    ln1_rstd: &'a mut [f32],  // (L, B, T)
    qkv: &'a mut [f32],       // (L, B, T, 3*C)
    atty: &'a mut [f32],      // (L, B, T, C)
    preatt: &'a mut [f32],    // (L, B, NH, T, T)
    att: &'a mut [f32],       // (L, B, NH, T, T)
    attproj: &'a mut [f32],   // (L, B, T, C)
    residual2: &'a mut [f32], // (L, B, T, C)
    ln2: &'a mut [f32],       // (L, B, T, C)
    ln2_mean: &'a mut [f32],  // (L, B, T)
    ln2_rstd: &'a mut [f32],  // (L, B, T)
    fch: &'a mut [f32],       // (L, B, T, 4*C)
    fch_gelu: &'a mut [f32],  // (L, B, T, 4*C)
    fcproj: &'a mut [f32],    // (L, B, T, C)
    residual3: &'a mut [f32], // (L, B, T, C)
    lnf: &'a mut [f32],       // (B, T, C)
    lnf_mean: &'a mut [f32],  // (B, T)
    lnf_rstd: &'a mut [f32],  // (B, T)
    logits: &'a mut [f32],    // (B, T, V)
    probs: &'a mut [f32],     // (B, T, V)
    losses: &'a mut [f32],    // (B, T)
}

impl<'a> ActivationTensors<'a> {
    fn new<'b>(
        acts_memory: &'b mut [f32],
        act_sizes: &[usize; NUM_ACTIVATION_TENSORS],
    ) -> ActivationTensors<'b> {
        let mut acts = Vec::with_capacity(NUM_ACTIVATION_TENSORS);
        let mut ptr = 0usize;
        for s in act_sizes.iter() {
            // acts.push(&mut acts_memory[ptr..ptr + s]);
            acts.push(unsafe {
                std::slice::from_raw_parts_mut(acts_memory.as_mut_ptr().offset(ptr as isize), *s)
            });
            ptr += s;
        }

        ActivationTensors {
            encoded: std::mem::take(&mut acts[0]),
            ln1: std::mem::take(&mut acts[1]),
            ln1_mean: std::mem::take(&mut acts[2]),
            ln1_rstd: std::mem::take(&mut acts[3]),
            qkv: std::mem::take(&mut acts[4]),
            atty: std::mem::take(&mut acts[5]),
            preatt: std::mem::take(&mut acts[6]),
            att: std::mem::take(&mut acts[7]),
            attproj: std::mem::take(&mut acts[8]),
            residual2: std::mem::take(&mut acts[9]),
            ln2: std::mem::take(&mut acts[10]),
            ln2_mean: std::mem::take(&mut acts[11]),
            ln2_rstd: std::mem::take(&mut acts[12]),
            fch: std::mem::take(&mut acts[13]),
            fch_gelu: std::mem::take(&mut acts[14]),
            fcproj: std::mem::take(&mut acts[15]),
            residual3: std::mem::take(&mut acts[16]),
            lnf: std::mem::take(&mut acts[17]),
            lnf_mean: std::mem::take(&mut acts[18]),
            lnf_rstd: std::mem::take(&mut acts[19]),
            logits: std::mem::take(&mut acts[20]),
            probs: std::mem::take(&mut acts[21]),
            losses: std::mem::take(&mut acts[22]),
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
    acts_memory: Vec<f32>,
    // gradients of the activations
    grads_acts: ActivationTensors<'a>,
    grads_acts_memory: Vec<f32>,
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
        let mut acts_memory = vec![0f32; num_activations];
        let acts_memory_slice =
            unsafe { std::slice::from_raw_parts_mut(acts_memory.as_mut_ptr(), acts_memory.len()) };
        let acts = ActivationTensors::new(acts_memory_slice, &act_sizes);

        // activations gradients
        let mut grads_acts_memory = vec![0f32; num_activations];
        let grads_acts_memory_slice = unsafe {
            std::slice::from_raw_parts_mut(grads_acts_memory.as_mut_ptr(), grads_acts_memory.len())
        };
        let grads_acts = ActivationTensors::new(grads_acts_memory_slice, &act_sizes);

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
            acts_memory,
            grads_acts,
            grads_acts_memory,
            seq_len: t as u32,
            batch_size: b as u32,
            inputs: vec![],
            targets: vec![],
            mean_loss: -1.0,
        }
    }

    fn forward(&mut self, inputs: &Vec<u32>, b: usize, t: usize) {
        // validate inputs, all indices must be in the range [0, V)
        for i in inputs.iter() {
            assert!(*i < self.config.vocab_size as u32);
        }

        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        // if b != self.batch_size as usize || t != self.seq_len as usize {
        //     panic!("invalid batch or sequence length");
        // }

        // convenience parameters
        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let c = self.config.channels;

        // cache the inputs
        self.inputs = inputs.clone();
        // forward pass
        let params = &self.params; // for brevity
        let acts = &mut self.acts;
        let mut residual: &[f32];
        encoder_forward(&mut acts.encoded, inputs, params.wte, params.wpe, b, t, c); // encoding goes into residual[0]
        for i_l in 0..l {
            // get the pointers of the weights for this layer
            let l_ln1w = &params.ln1w[i_l * c..];
            let l_ln1b = &params.ln1b[i_l * c..];
            let l_qkvw = &params.qkvw[i_l * 3 * c * c..];
            let l_qkvb = &params.qkvb[i_l * 3 * c..];
            let l_attprojw = &params.attprojw[i_l * c * c..];
            let l_attprojb = &params.attprojb[i_l * c..];
            let l_ln2w = &params.ln2w[i_l * c..];
            let l_ln2b = &params.ln2b[i_l * c..];
            let l_fcw = &params.fcw[i_l * 4 * c * c..];
            let l_fcb = &params.fcb[i_l * 4 * c..];
            let l_fcprojw = &params.fcprojw[i_l * c * 4 * c..];
            let l_fcprojb = &params.fcprojb[i_l * c..];

            // get the pointers of the activations for this layer
            let l_ln1 = &mut acts.ln1[i_l * b * t * c..];
            let l_ln1_mean = &mut acts.ln1_mean[i_l * b * t..];
            let l_ln1_rstd = &mut acts.ln1_rstd[i_l * b * t..];
            let l_qkv = &mut acts.qkv[i_l * b * t * 3 * c..];
            let l_atty = &mut acts.atty[i_l * b * t * c..];
            let l_preatt = &mut acts.preatt[i_l * b * nh * t * t..];
            let l_att = &mut acts.att[i_l * b * nh * t * t..];
            let l_attproj = &mut acts.attproj[i_l * b * t * c..];
            let l_residual2 = &mut acts.residual2[i_l * b * t * c..];
            let l_ln2 = &mut acts.ln2[i_l * b * t * c..];
            let l_ln2_mean = &mut acts.ln2_mean[i_l * b * t..];
            let l_ln2_rstd = &mut acts.ln2_rstd[i_l * b * t..];
            let l_fch = &mut acts.fch[i_l * b * t * 4 * c..];
            let l_fch_gelu = &mut acts.fch_gelu[i_l * b * t * 4 * c..];
            let l_fcproj = &mut acts.fcproj[i_l * b * t * c..];

            {
                residual = if i_l == 0 {
                    acts.encoded
                } else {
                    &acts.residual3[(i_l - 1) * b * t * c..]
                };

                // now do the forward pass
                layernorm_forward(
                    l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, b, t, c,
                );
                matmul_forward(l_qkv, l_ln1, l_qkvw, Some(l_qkvb), b, t, c, 3 * c);
                attention_forward(l_atty, l_preatt, l_att, l_qkv, b, t, c, nh);
                matmul_forward(l_attproj, l_atty, l_attprojw, Some(l_attprojb), b, t, c, c);
                residual_forward(l_residual2, residual, l_attproj, b * t * c);
                layernorm_forward(
                    l_ln2,
                    l_ln2_mean,
                    l_ln2_rstd,
                    l_residual2,
                    l_ln2w,
                    l_ln2b,
                    b,
                    t,
                    c,
                );
            }
            matmul_forward(l_fch, l_ln2, l_fcw, Some(l_fcb), b, t, c, 4 * c);
            gelu_forward(l_fch_gelu, l_fch, b * t * 4 * c);
            matmul_forward(
                l_fcproj,
                l_fch_gelu,
                l_fcprojw,
                Some(l_fcprojb),
                b,
                t,
                4 * c,
                c,
            );
            {
                let l_residual3 = &mut acts.residual3[i_l * b * t * c..];
                residual_forward(l_residual3, l_residual2, l_fcproj, b * t * c);
            }
        }
        residual = &acts.residual3[(l - 1) * b * t * c..]; // last residual is in residual3
        layernorm_forward(
            &mut acts.lnf,
            &mut acts.lnf_mean,
            &mut acts.lnf_rstd,
            residual,
            params.lnfw,
            params.lnfb,
            b,
            t,
            c,
        );
        matmul_forward(
            &mut acts.logits,
            &mut acts.lnf,
            params.wte,
            None,
            b,
            t,
            c,
            v,
        );
        softmax_forward(&mut acts.probs, &mut acts.logits, b, t, v);
    }
}

fn encoder_forward(
    out: &mut [f32],
    inp: &Vec<u32>,
    wte: &[f32],
    wpe: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for i_b in 0..b {
        for i_t in 0..t {
            // seek to the output position in out[b,t,:]
            let out_bt = &mut out[i_b * t * c + i_t * c..];
            // get the index of the token at inp[b, t]
            let ix = inp[i_b * t + i_t] as usize;
            // seek to the position in wte corresponding to the token
            let dwte_ix = &wte[ix * c..];
            // seek to the position in wpe corresponding to the position
            let dwpe_t = &wpe[i_t * c..];
            // add the two vectors and store the result in out[b,t,:]
            for i in 0usize..c {
                out_bt[i] = dwte_ix[i] + dwpe_t[i];
            }
        }
    }
}

fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    let eps = 1e-5_f32;
    for i_b in 0..b {
        for i_t in 0..t {
            // seek to the input position inp[b,t,:]
            let x = &inp[i_b * t * c + i_t * c..];
            // calculate the mean
            let mut m = 0f32;
            for i in 0..c {
                m += x[i];
            }
            m = m / c as f32;
            // calculate the variance (without any bias correction)
            let mut v = 0f32;
            for i in 0..c {
                let xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / c as f32;
            // calculate the rstd (reciprocal standard deviation)
            let s = 1.0 / f32::sqrt(v + eps);
            // seek to the output position in out[b,t,:]
            let out_bt = &mut out[i_b * t * c + i_t * c..];
            for i in 0..c {
                let n = s * (x[i] - m); // normalize
                let o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[i_b * t + i_t] = m;
            rstd[i_b * t + i_t] = s;
        }
    }
}

fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    for i_b in 0..b {
        for i_t in 0..t {
            let out_bt = &mut out[i_b * t * oc + i_t * oc..];
            let inp_bt = &inp[i_b * t * c + i_t * c..];
            for o in 0..oc {
                let mut val = match bias {
                    Some(b) => b[o],
                    None => 0f32,
                };
                let wrow = &weight[o * c..];
                for i in 0..c {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    let c3 = c * 3;
    let hs = c / nh; // head size
    let scale = 1.0 / f32::sqrt(hs as f32);

    for i_b in 0..b {
        for i_t in 0..t {
            for h in 0..nh {
                let query_t = &inp[i_b * t * c3 + i_t * c3 + h * hs..];
                let preatt_bth = &mut preatt[i_b * nh * t * t + h * t * t + i_t * t..];
                let att_bth = &mut att[i_b * nh * t * t + h * t * t + i_t * t..];

                // pass 1: calculate query dot key and maxval
                let mut maxval = -10000f32; // TODO something better
                for t2 in 0..i_t {
                    let key_t2 = &inp[i_b * t * c3 + t2 * c3 + h * hs + c..]; // +C because it's key

                    // (query_t) dot (key_t2)
                    let mut val = 0f32;
                    for i in 0..hs {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                let mut expsum = 0f32;
                for t2 in 0..i_t {
                    let expv = f32::exp(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                let expsum_inv = if expsum == 0f32 { 0f32 } else { 1f32 / expsum };

                // pass 3: normalize to get the softmax
                for t2 in 0..t {
                    if t2 <= i_t {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0f32;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                let out_bth = &mut out[i_b * t * c + i_t * c + h * hs..];
                for t2 in 0..i_t {
                    let value_t2 = &inp[i_b * t * c3 + t2 * c3 + h * hs + c * 2..]; // +C*2 because it's value
                    let att_btht2 = att_bth[t2];
                    for i in 0..hs {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

fn gelu_forward(out: &mut [f32], inp: &[f32], n: usize) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for i in 0..n {
        let x = inp[i];
        let cube = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + f32::tanh(GELU_SCALING_FACTOR * (x + cube)));
    }
}

fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], n: usize) {
    for i in 0..n {
        out[i] = inp1[i] + inp2[i];
    }
}

fn softmax_forward(probs: &mut [f32], logits: &[f32], b: usize, t: usize, v: usize) {
    // output: probs are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,V) of the unnormalized log probabilities
    for i_b in 0..b {
        for i_t in 0..t {
            // probs <- softmax(logits)
            let logits_bt = &logits[i_b * t * v + i_t * v..];
            let probs_bt = &mut probs[i_b * t * v + i_t * v..];

            // maxval is only calculated and subtracted for numerical stability
            let mut maxval = -10000f32; // TODO something better
            for i in 0..v {
                if logits_bt[i] > maxval {
                    maxval = logits_bt[i];
                }
            }
            let mut sum = 0f32;
            for i in 0..v {
                probs_bt[i] = f32::exp(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            for i in 0..v {
                probs_bt[i] /= sum;
            }
        }
    }
}

fn sample_mult(probabilities: &[f32], n: usize, coin: f32) -> usize {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    let mut cdf = 0f32;
    for i in 0..n {
        cdf += probabilities[i];
        if coin < cdf {
            return i;
        }
    }
    n - 1 // in case of rounding errors
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

    #[test]
    fn test_gpt2_forward() {
        let tokenizer = Tokenizer::new("gpt2_tokenizer.bin");
        let mut gpt2 = GPT2::from_checkpoint("gpt2_124M.bin", B, T);
        let inputs = vec![
            9203u32, 262, 39418, 13, 628, 50256, 22747, 49208, 805, 25, 198, 9203, 262, 39418, 0,
            628, 50256, 44879, 40, 3535, 1565, 2937, 25, 198, 42012, 13, 628, 50256, 22747, 49208,
            805, 25, 198, 8496, 338, 326, 30, 628, 50256,
        ];
        gpt2.forward(&inputs, B, T);
        let probs = gpt2.acts.probs;
        let coin = rand::random::<f32>();
        let next_token = sample_mult(probs, gpt2.config.vocab_size, coin);
        let token_str = tokenizer.decode(next_token as u32).unwrap();
        println!("{}", token_str);
    }
}
