/*!
This example shows how to read glove vectors and build a Granne index.

# Example

```
cargo run --release --example glove glove.6B.50d.txt
```

# Input data

Input data needs to be downloaded and unzipped from:

https://nlp.stanford.edu/data/wordvecs/glove.6B.zip

This data is made available under the Public Domain Dedication and License v1.0 whose
full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/
*/

use granne::{self, Builder};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn parse_line(line: &str) -> std::io::Result<(String, granne::angular::Vector<'static>)> {
    let mut line_iter = line.split_whitespace();
    let token = line_iter.next().ok_or(std::io::ErrorKind::InvalidData)?;
    let vec: granne::angular::Vector = line_iter.map(|d| d.parse::<f32>().unwrap()).collect();

    Ok((token.to_string(), vec))
}

fn main() -> std::io::Result<()> {
    let input_file: String = std::env::args().skip(1).next().expect("Missing input_file!");
    let file = BufReader::new(File::open(input_file)?);

    // reading the input data
    let mut elements = granne::angular::Vectors::new();
    let mut tokens = Vec::new();
    for line in file.lines() {
        let (token, vector) = parse_line(&line?)?;

        tokens.push(token);
        elements.push(&vector);
    }

    // building the index
    let build_config = granne::BuildConfig::default().show_progress(true).max_search(10); // increase this for better results

    let mut builder = granne::GranneBuilder::new(build_config, elements);

    builder.build();

    // querying
    let index = builder.get_index();

    for &i in &[0, 134, 5555, 37000] {
        let res = index.search(&index.get_element(i), 200, 10);

        let res: Vec<_> = res.into_iter().map(|(j, d)| (&tokens[j], d)).collect();

        println!("\nThe closest words to \"{}\" are: \n{:?}", &tokens[i], res);
    }

    Ok(())
}
