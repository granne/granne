use std::io::{BufRead, BufReader, Result, Write};
use std::fs::File;
use std::path;
use memmap::Mmap;

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

pub fn read<P>(path: P, number: usize) -> Result<(Vec<FloatElement>, Vec<String>)>
where
    P: AsRef<path::Path>,
{
    let file = File::open(path)?;
    let file = BufReader::new(file);

    let mut elements = Vec::new();
    let mut words = Vec::new();

    for (word, element) in file.lines().map(|line| read_line(&line.unwrap())).take(
        number,
    )
    {
        elements.push(element);
        words.push(word);
    }

    return Ok((elements, words));

    //    return Ok(file.lines().map(|line| read_line(&line.unwrap()).1).collect());
}

fn write<T, B: Write>(vectors: &[T], buffer: &mut B) -> Result<()> {
    let data = unsafe {
        ::std::slice::from_raw_parts(
            vectors.as_ptr() as *const u8,
            vectors.len() * ::std::mem::size_of::<T>(),
        )
    };

    buffer.write(data)?;

    Ok(())
}

pub fn save_to_disk<T>(vectors: &[T], path: &str) -> Result<()> {
    let mut file = File::create(path)?;

    write(vectors, &mut file)
}


pub fn load_from_disk<T: Clone>(path: &str) -> Result<(Vec<T>)> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let vectors = load::<T>(&mmap);

    Ok(vectors.into())
}


pub fn load<T>(buffer: &[u8]) -> &[T] {
    let vectors: &[T] = unsafe {
        ::std::slice::from_raw_parts(
            buffer.as_ptr() as *const T,
            buffer.len() / ::std::mem::size_of::<T>(),
        )
    };

    vectors
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_file() {
        let data = read("/Users/erik/data/glove.6B/glove.1K.50d.txt", 1000);

        if let Ok((data, strings)) = data {
            assert_eq!(1000, data.len());
        } else {
            panic!("Could not read file");
        }
    }

    #[test]
    #[should_panic]
    fn read_nonexistent_file() {
        read("non_existent", 1000).unwrap();
    }
}
