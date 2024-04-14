use std::{fs::File, path::Path};

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
    const B: u32 = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    const T: u32 = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
}

struct DataLoader<'a> {
    // huperparameters
    b: u32, // batch size
    t: u32, // sequence length
    // input handling and its state
    tokens_file: &'a File,
    file_size: u32,
    current_position: u32,
    // output memory
    batch: &'a Vec<i32>,
    inputs: &'a Vec<i32>,
    targets: &'a Vec<i32>,
    // convenience variables
    num_batches: u32,
}
