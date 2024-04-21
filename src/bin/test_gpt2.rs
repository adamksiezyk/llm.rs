use std::{collections::HashMap, fs::File, io::Read, mem::size_of, path::Path, usize};

// hyperparameters
const B: usize = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
const T: usize = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2

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
    let sentence = train_loader
        .next_batch()
        .iter()
        .filter_map(|t| tokenizer.decode(*t).ok())
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join("");
    println!("{sentence}");
}

fn read_u32(file: &mut File, size: usize) -> Vec<u32> {
    let mut buffer = vec![0u8; size * size_of::<u32>()];
    file.read(&mut buffer).unwrap();

    buffer
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
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
