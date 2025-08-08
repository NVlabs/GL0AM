#![allow(non_snake_case)]

use saif_dumper::compareSaifFiles;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
 // Get the SAIF file paths and tolerance from command line arguments
 let args: Vec<String> = env::args().collect();
 if args.len() != 5 {
  eprintln!("Usage: {} <golden_saif_file> <compare_saif_file> <tolerance> <TCtolerance>", args[0]);
  eprintln!("Example: {} /path/to/golden.saif /path/to/compare.saif 0 0", args[0]);
  eprintln!("Example: {} /path/to/golden.saif /path/to/compare.saif 5 10", args[0]);
  eprintln!("Note: tolerance=0 for exact T0/T1 match, tolerance>0 for approximate T0/T1 match");
  eprintln!("Note: TCtolerance=0 for exact TC match, TCtolerance>0 for absolute difference TC match");
  return Err("Invalid number of arguments".into());
 }

 let golden_file_path = &args[1];
 let compare_file_path = &args[2];
 
 // Parse tolerance parameters
 let tolerance: u32 = args[3].parse().map_err(|_| {
  eprintln!("Error: tolerance must be a valid unsigned integer");
  "Invalid tolerance value"
 })?;
 
 let TCtolerance: u32 = args[4].parse().map_err(|_| {
  eprintln!("Error: TCtolerance must be a valid unsigned integer");
  "Invalid TCtolerance value"
 })?;

 // Test the compareSaifFiles function
 match compareSaifFiles(golden_file_path, compare_file_path, tolerance, TCtolerance) {
  Ok(()) => {
   // Success message is handled by compareSaifFiles function
  }
  Err(e) => {
   eprintln!("❌ ERROR: {}", e);
   return Err(e);
  }
 }
 Ok(())
}
