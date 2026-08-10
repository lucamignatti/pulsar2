pub mod cpp_records;
mod data_reader;
pub mod tick_record;

use std::io::ErrorKind;

use crate::rl_comparison_test::recording::tick_record::TickRecord;
use cpp_records::*;
use data_reader::DataReader;

const RLPR_MAGIC_BYTES: [u8; 4] = [82, 76, 80, 82];
const RLPR_VERSION: u32 = 2;
const RLPR_MAX_CARS: usize = 2;

#[allow(dead_code)]
pub struct Recording {
    pub name: String,
    pub info: RecordingInfo,
    pub ticks: Vec<TickRecord>,
}

impl Recording {
    pub fn from_bytes(name: &str, bytes: &[u8]) -> Result<Recording, std::io::Error> {
        let mut reader = DataReader::new(bytes);

        for magic_byte in RLPR_MAGIC_BYTES {
            if reader.read_u8()? != magic_byte {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    "File is not a valid recording (wrong magic)",
                ));
            }
        }

        let are_we_big_endian = cfg!(target_endian = "big");
        let is_file_big_endian = reader.read_bool()?;
        if is_file_big_endian != are_we_big_endian {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "File has wrong endianness",
            ));
        }

        let version = reader.read_u32()?;
        if version != RLPR_VERSION {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("RLPR Version mismatch (expected: {RLPR_VERSION}, got: {version})"),
            ));
        }

        let info = unsafe { reader.read_struct_unsafe::<RecordingInfo>() }?;
        let num_cars = info.num_cars as usize;
        if num_cars > RLPR_MAX_CARS {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!("RLPR recording has too many cars (max: {RLPR_MAX_CARS}, got: {num_cars})"),
            ));
        }

        let num_ticks = reader.read_u32()?;
        let mut ticks = Vec::with_capacity(num_ticks as usize);

        let car_size = size_of::<CarRecord>() as u32;
        let ball_size = size_of::<PhysRecord>() as u32;
        for _ in 0..num_ticks {
            // Demolished cars are OMITTED from ticks while they are gone (RLRecord2
            // spec), so a tick holds 0..=num_cars CarRecords followed by exactly one
            // ball PhysRecord. The size prefixes disambiguate (584 vs 332).
            let mut car_records = Vec::with_capacity(num_cars);
            loop {
                let next_size = reader.peek_u32()?;
                if next_size == car_size && car_records.len() < num_cars {
                    let car_record = unsafe { reader.read_struct_unsafe::<CarRecord>() }?;
                    car_records.push(car_record);
                } else if next_size == ball_size {
                    break;
                } else {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        format!("Unexpected struct size prefix {next_size} in tick stream"),
                    ));
                }
            }
            let ball_record = unsafe { reader.read_struct_unsafe::<PhysRecord>() }?;
            ticks.push(TickRecord {
                car_records,
                ball_record,
            });
        }

        if reader.num_bytes_left() > 0 {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "RLPR recording still has {} bytes left after reading all ticks",
                    reader.num_bytes_left()
                ),
            ));
        }

        Ok(Self {
            name: name.to_string(),
            info,
            ticks,
        })
    }
}
