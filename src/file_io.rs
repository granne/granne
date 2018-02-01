use std::io::{BufRead, BufReader, Result, Write, Read};
use std::fs::File;
use std::str::FromStr;
use std::path;
use memmap::Mmap;
use serde_json;
use types::{FloatElement, Int8Element, DIM, AngularVector};
use rayon::prelude::*;
use std::iter::FromIterator;


fn read_line_generic<T: FromIterator<F>, F: FromStr>(line: &str) -> (String, T) {
    let mut iter = line.split_whitespace();

    let word = String::from(iter.next().unwrap());

    let element: T =
        iter.map(|e| {
            if let Ok(value) = e.parse::<F>() {
                value
            } else {
                panic!("Could not convert to number");
            }
        })
        .collect();

    return (word, element);
}

fn read_line_glove(line: &str) -> (String, FloatElement) {
    let mut iter = line.split_whitespace();

    let word = String::from(iter.next().unwrap());

    let mut v = [0.0f32; DIM];
    for (i, e) in iter.enumerate() {
        v[i] = e.parse::<f32>().unwrap();
    }

    return (word, v.into());
}

fn read_line(line: &str) -> (String, FloatElement) {
    let mut parts = line.split('\t');

    let id: String = parts.next().unwrap().into();

    let mut va = [0.0f32; DIM];

    if let Ok(v) = serde_json::from_str::<Vec<f32>>(parts.next().unwrap()) {
        assert_eq!(DIM, v.len());

        va.copy_from_slice(v.as_slice());
    }

    (id, va.into())
}

fn read_int_line(line: &str) -> (String, Int8Element) {
    let mut parts = line.split('\t');

    let id: String = parts.next().unwrap().into();

    let mut va = [0i8; DIM];

    if let Ok(v) = serde_json::from_str::<Vec<i8>>(parts.next().unwrap()) {
        assert_eq!(DIM, v.len());

        va.copy_from_slice(v.as_slice());
    }

    (id, va.into())
}

pub fn read<P>(path: P, number: usize) -> Result<(Vec<FloatElement>, Vec<String>)>
where
    P: AsRef<path::Path>,
{
    let file = File::open(path)?;
    let file = BufReader::new(file);

    let mut elements = Vec::new();
    let mut words = Vec::new();

    for (word, element) in file.lines().map(|line| read_line_glove(&line.unwrap())).take(
        number,
    )
    {
        elements.push(element);
        words.push(word);
    }

    elements.shrink_to_fit();
    words.shrink_to_fit();

    return Ok((elements, words));
}

pub fn read_int<P>(path: P, number: usize) -> Result<(Vec<Int8Element>, Vec<String>)>
where
    P: AsRef<path::Path>,
{
    let file = File::open(path)?;
    let file = BufReader::new(file);

    let mut elements = Vec::new();
    let mut words = Vec::new();

    for (word, element) in file.lines().map(|line| read_int_line(&line.unwrap())).take(
        number,
    )
    {
        elements.push(element);
        words.push(word);

        if words.len() % 10_000_000 == 0 {
            println!("Added {} vectors", words.len());
        }
    }

    elements.shrink_to_fit();
    words.shrink_to_fit();

    return Ok((elements, words));
}

pub fn write<T, B: Write>(vectors: &[T], buffer: &mut B) -> Result<()> {
    let data = unsafe {
        ::std::slice::from_raw_parts(
            vectors.as_ptr() as *const u8,
            vectors.len() * ::std::mem::size_of::<T>(),
        )
    };

    buffer.write_all(data)
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

pub fn read_elements<T : Clone, B: Read>(reader: &mut B) -> Result<Vec<T>> {
    use std::mem::size_of;

    const BUFFER_SIZE: usize = 512;
    let mut buffer = [0u8; BUFFER_SIZE];

    let mut elements: Vec<T> = Vec::new();

    while reader.read_exact(&mut buffer[..size_of::<T>()]).is_ok() {

        elements.push(unsafe { (*(&buffer[0] as *const u8 as *const T)).clone() })
    }

    return Ok(elements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_file() {
        let data = read("example_data/glove.1K.100d.txt", 1000);

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
