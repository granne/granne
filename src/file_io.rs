use std::io::{BufRead, BufReader, Result};
use std::fs::File;
use std::path;

use types::{FloatElement, DIM};


fn read_line(line: &str) -> (String, FloatElement) {
    let mut iter = line.split_whitespace();
    
    let word = String::from(iter.next().unwrap());

    let mut v = [0.0f32; DIM];
    for (i, e) in iter.enumerate() {
        v[i] = e.parse::<f32>().unwrap();
    }

    return (word, v.into());
}

pub fn read<P>(path : P, number: usize) -> Result<(Vec<FloatElement>, Vec<String>)>
    where P : AsRef<path::Path> {
    let file = File::open(path)?;
    let file = BufReader::new(file);
    
    let mut elements = Vec::new();
    let mut words = Vec::new();

    for (word, element) in file.lines()
        .map(|line| read_line(&line.unwrap()))
        .take(number) 
    {
        elements.push(element);
        words.push(word);
    }
    
    return Ok((elements, words));                                        

//    return Ok(file.lines().map(|line| read_line(&line.unwrap()).1).collect());
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn read_file() {
        let data = read("/Users/erik/data/glove.6B/glove.1K.50d.txt", 1000);
        
        if let Ok((data, strings)) = data {
            assert_eq!(1000, data.len());
        }
        else {
            panic!("Could not read file");
        }
    }

    #[test]
    #[should_panic]
    fn read_nonexistent_file() {
        read("non_existent", 1000).unwrap();
    }
}
