use std::io::{BufRead, BufReader, Result, Write, Read};
use std::fs::File;
use std::str::FromStr;
use std::path;
use memmap::Mmap;
use std::iter::FromIterator;

fn read_line<T: FromIterator<F>, F: FromStr>(line: &str) -> (String, T) {
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


pub fn read<P, T, F>(path: P) -> Result<(Vec<T>, Vec<String>)>
where
    P: AsRef<path::Path>,
    T: FromIterator<F>,
    F: FromStr
{
    let file = File::open(path)?;
    let file = BufReader::new(file);

    let mut elements = Vec::new();
    let mut words = Vec::new();

    for (word, element) in file
        .lines()
        .map(|line| read_line::<T, F>(&line.unwrap()))
    {
        elements.push(element);
        words.push(word);
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

pub fn read_elements<T : Clone, B: Read>(reader: &mut B, max_number_of_elements: usize) -> Result<Vec<T>> {
    use std::mem::size_of;

    const BUFFER_SIZE: usize = 512;
    let mut buffer = [0u8; BUFFER_SIZE];

    let mut elements: Vec<T> = Vec::new();

    while reader.read_exact(&mut buffer[..size_of::<T>()]).is_ok() && elements.len() < max_number_of_elements {

        elements.push(unsafe { (*(&buffer[0] as *const u8 as *const T)).clone() })
    }

    elements.shrink_to_fit();

    return Ok(elements)
}

pub fn read_f32<P: AsRef<path::Path>>(path: P) -> Result<(Vec<f32>, Vec<String>)>
{
    let file = File::open(path)?;
    let file = BufReader::new(file);

    let mut words = Vec::new();
    let mut element_data = Vec::new();
    let mut dimension = 0;

    for (word, components) in file
        .lines()
        .map(|line| {
            let line = line.unwrap();
            let mut iter = line.split_whitespace();
            let word = String::from(iter.next().unwrap());

            let components: Vec<f32> =
                iter.map(|e| {
                    if let Ok(value) = e.parse::<f32>() {
                        value
                    } else {
                        panic!("Could not convert to number");
                    }
                }).collect();

            if dimension == 0 {
                dimension = components.len();
            } else {
                assert_eq!(dimension, components.len());
            }

            (word, components)
        })
    {
        element_data.extend(components);
        words.push(word);
    }

    return Ok((element_data, words));
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AngularVector, ComparableTo};

    #[test]
    fn read_file() {
        let data = read("example_data/test.1K.100d.txt");

        if let Ok((data, _strings)) = data {
            assert_eq!(1000, data.len());

            for i in 0..data.len() {
                let v: &AngularVector = &data[i];
                assert!(v.dist(v).into_inner() < 0.01f32);
            }
        } else {
            panic!("Could not read file");
        }
    }

    #[test]
    #[should_panic]
    fn read_nonexistent_file() {
        let _: (Vec<AngularVector>, _) = read("non_existent").unwrap();
    }
}
