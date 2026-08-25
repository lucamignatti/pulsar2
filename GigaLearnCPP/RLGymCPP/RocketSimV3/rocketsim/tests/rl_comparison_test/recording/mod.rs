pub mod cpp_records;
mod data_reader;
pub mod tick_record;

use std::io::ErrorKind;
use std::mem::size_of;

use crate::rl_comparison_test::recording::tick_record::TickRecord;
use cpp_records::*;
use data_reader::DataReader;

const RLPR_MAGIC_BYTES: [u8; 4] = [82, 76, 80, 82];
const RLPR_VERSION_MIN: u32 = 2;
const RLPR_VERSION_MAX: u32 = 5;
const RLPR_MAX_CARS: usize = 4;

#[allow(dead_code)]
pub struct Recording {
    pub name: String,
    pub version: u32,
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
        if !(RLPR_VERSION_MIN..=RLPR_VERSION_MAX).contains(&version) {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "RLPR version {version} unsupported (this reader handles {RLPR_VERSION_MIN}..={RLPR_VERSION_MAX})"
                ),
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

        let car_size_v5 = size_of::<CarRecord>() as u32;
        let car_size_v4 = size_of::<CarRecordV4>() as u32;
        let car_size_v3 = size_of::<CarRecordV3>() as u32;
        let car_size_v2 = size_of::<CarRecordV2>() as u32;
        let ball_size_v5 = size_of::<PhysRecord>() as u32;
        let ball_size_v4 = size_of::<PhysRecordV4>() as u32;
        for tick_idx in 0..num_ticks {
            // Demolished cars are omitted while gone. Size prefixes tell a CarRecord
            // (1232 / 864 / 744 / 584) from the ball PhysRecord (344 / 332).
            let mut car_records = Vec::with_capacity(num_cars);
            let ball_record = loop {
                let next_size = reader.peek_u32()?;
                if next_size == car_size_v5
                    || next_size == car_size_v4
                    || next_size == car_size_v3
                    || next_size == car_size_v2
                {
                    if car_records.len() >= num_cars {
                        return Err(std::io::Error::new(
                            ErrorKind::InvalidData,
                            format!(
                                "RLPR tick {tick_idx} has more than the declared {num_cars} cars"
                            ),
                        ));
                    }
                    let car_record = if next_size == car_size_v5 {
                        unsafe { reader.read_struct_unsafe::<CarRecord>() }?
                    } else if next_size == car_size_v4 {
                        unsafe { reader.read_struct_unsafe::<CarRecordV4>() }?.into()
                    } else if next_size == car_size_v3 {
                        unsafe { reader.read_struct_unsafe::<CarRecordV3>() }?.into()
                    } else {
                        unsafe { reader.read_struct_unsafe::<CarRecordV2>() }?.into()
                    };
                    car_records.push(car_record);
                } else if next_size == ball_size_v5 {
                    break unsafe { reader.read_struct_unsafe::<PhysRecord>() }?;
                } else if next_size == ball_size_v4 {
                    break unsafe { reader.read_struct_unsafe::<PhysRecordV4>() }?.into();
                } else {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        format!("Unexpected struct size prefix {next_size} in tick stream"),
                    ));
                }
            };
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
            version,
            info,
            ticks,
        })
    }
}
