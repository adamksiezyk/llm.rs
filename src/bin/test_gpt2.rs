use std::{fs::File, io::Read, mem::size_of, path::Path, usize};

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
    dbg!(tokenizer.token_table);
    let mut train_loader = DataLoader::new(train_tokens.to_str().unwrap());
    let batch = train_loader.next_batch();
    dbg!(batch);
}

trait FromBytes {
    fn from_le_bytes(buff: [u8; 4]) -> Self;
}

fn buffer_to_type<R: FromBytes, const SIZE: usize, const COUNT: usize>(
    buff: &[u8],
    out: &mut [R; COUNT],
) {
    for (i, chunk) in buff.chunks(SIZE).enumerate() {
        out[i] = R::from_le_bytes(chunk.try_into().unwrap());
    }
}

struct DataLoader {
    // input handling and its state
    tokens_file: File,
    file_size: usize,
    current_position: usize,
    // output memory
    batch: [i32; B * T + 1],
    // convenience variables
    num_batches: usize,
}

impl DataLoader {
    fn new(filename: &str) -> DataLoader {
        let file = File::open(filename).expect(&format!("Could not open file: '{filename}'"));
        let file_size = file.metadata().expect("Culd not read file metadata").len() as usize;

        if file_size < (B * T + 1) * size_of::<i32>() {
            panic!("Error: file size is too small for the batch size and sequence length");
        }

        DataLoader {
            tokens_file: file,
            file_size,
            current_position: 0,
            num_batches: file_size / (B * T * size_of::<i32>()),
            batch: [0; B * T + 1],
        }
    }

    fn reset(&mut self) {
        self.current_position = 0;
    }

    fn next_batch<'a>(&'a mut self) -> &'a [i32; B * T + 1] {
        // if we are at the end of the file, loop back to the beginning
        if self.current_position + (B * T + 1) * size_of::<i32>() > self.file_size {
            self.current_position = 0;
        }

        let mut buf = [0u8; (B * T + 1) * size_of::<i32>()];
        self.tokens_file
            .read(&mut buf)
            .expect("Error while reading batch from file");
        for i in 0..(B * T + 1) {
            self.batch[i] =
                i32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]);
        }
        &self.batch
    }
}

struct Tokenizer {
    vocab_size: usize,
    token_table: Vec<String>,
}

impl Tokenizer {
    fn new(filename: &str) -> Tokenizer {
        let mut file = File::open(filename).expect(&format!("Could not open file: '{filename}'"));

        // read in the header
        let mut header_bytes = [0u8; 1024];
        file.read(&mut header_bytes);
        let mut header = [0u32; 256];
        for i in 0..256 {
            header[i] = u32::from_le_bytes([
                header_bytes[i * 4],
                header_bytes[i * 4 + 1],
                header_bytes[i * 4 + 2],
                header_bytes[i * 4 + 3],
            ]);
        }
        assert!(header[0] == 20240328);
        assert!(header[1] == 1);
        let vocab_size = header[2] as usize;
        // read in all the tokens
        let mut length = [0u8; 1];
        let mut token_table = Vec::<String>::new();
        for i in 0..vocab_size {
            file.read(&mut length);
            let mut buff = vec![0u8; length[0] as usize];
            file.read(&mut buff);
            // let mut null_termiator = [0u8; 1];
            // '\0'.encode_utf8(&mut null_termiator);
            // buff.extend(null_termiator);
            dbg!(&buff);
            // TODO change to UTF-16
            let token = String::from_utf8(buff).unwrap();
            token_table.insert(i, token);
        }

        Tokenizer {
            vocab_size,
            token_table,
        }
    }
}
