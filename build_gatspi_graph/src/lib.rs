use netlistdb::*;
use netlistdb::{Direction, LeafPinProvider};
use compact_str::CompactString;
use sverilogparse::SVerilogRange;
use ulib::UVec;
use ulib::Device;
use regex::Regex;
use lazy_static::lazy_static;
use itertools::Itertools;
use std::iter;
use rayon::prelude::*;
use std::sync::atomic::{AtomicPtr, Ordering};

use std::collections::HashMap;


pub mod stdlib_attributes;
pub use stdlib_attributes::{MLCADDesignContest2025StdLib,StandardCellTypeAttribute};

#[derive(Debug, Clone,serde::Serialize)]
pub struct GATSPIGraph {
 pub gatspi_cells: Vec<usize>,
 pub gatspi_celltypes: Vec<u16>,
 pub start: Vec<usize>,
 pub items: Vec<usize>,
 pub values: Vec<u8>,
 pub pinid2gatspiid: Vec<usize>,
 pub num_of_gatspi_cells: usize,
 //some debug stuff
 pub num_of_top_ports: usize,
 pub gatspi_cellname_index: Vec<(String,String)>,
 pub gatspi_port_index: HashMap<usize, String>,
}

lazy_static! {
 static ref SEQ_REGEX: Regex = Regex::new(r".*(DFF|DHL|DLL|ICG|SDF|sram_).*").unwrap();
}


pub struct StdCellPinDefs();

impl LeafPinProvider for StdCellPinDefs {
 fn direction_of(
  &self,
  macro_name: &CompactString,
  pin_name: &CompactString, pin_idx: Option<isize>
  ) -> Direction {
  if let true = SEQ_REGEX.is_match(macro_name.as_str()) {
   match pin_name.as_str() {
    "GCLK" | "Q" | "QN" | "rd_out" => Direction::O,
    "D" | "SE" | "SI" | "CLK" | "RESETN" | "RESET" | "SETN" | "SET" | "ENA" | "addr_in" | "ce_in" | "clk" | "wd_in" | "we_in" => Direction::Unknown,
    _ => { 
     use netlistdb::{GeneralPinName, HierName};
     panic!("Cannot recognize sequential pin type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
     (HierName::single(macro_name.clone()),
     pin_name, pin_idx).dbg_fmt_pin());
    }
   }
  } else {
   match pin_name.as_str() {
    "Y" | "CON" | "SN"  => Direction::O,
    "A1" | "A2" | "A3" | "B1" | "B2" | "B3" | "C1" | "C2" | "C3" | "A" | "B" | "C" | "D" | "E" | "CI" => Direction::I,
    _ => { 
     use netlistdb::{GeneralPinName, HierName};
     panic!("Cannot recognize unknown pin type {}, please make sure the verilog netlist is synthesized from Contest tech lib.",
     (HierName::single(macro_name.clone()),
     pin_name, pin_idx).dbg_fmt_pin());
    }
   }
  }
 } 
    
 fn width_of(
  &self,
  _macro_name: &CompactString,
  _pin_name: &CompactString
 ) -> Option<SVerilogRange> {
  None
 }
}



impl GATSPIGraph {

 pub fn build_graph(db: &NetlistDB, stdlib_info: &(impl StandardCellTypeAttribute + std::marker::Sync) ) -> Self {
  let num_ports = db.cell2pin.start[1];
  let max_threads = rayon::current_num_threads(); println!("Rayon detects {} threads available", max_threads);
  let range_size = db.num_pins - num_ports; let parallel_stride = (range_size + max_threads - 1) / max_threads;
  let mut these_opins: Vec<usize> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
   let start = num_ports + thread_id * parallel_stride;
   let end = if thread_id == max_threads - 1 { db.num_pins } else { (num_ports + (thread_id + 1) * parallel_stride).min(db.num_pins) };
   (start..end).filter(|&index| db.pindirect[index] == Direction::O).collect::<Vec<usize>>()
  }).collect();
  let legit_top_ports = Vec::from_iter( (0..num_ports).filter( |pin| (Some(db.pin2net[*pin]) != db.net_zero) && (Some(db.pin2net[*pin]) != db.net_one) && (db.pindirect[*pin] == Direction::O) ) );    
  these_opins.splice(0..0, legit_top_ports.clone()); 
  let these_opins2 = Vec::from(these_opins);
  let stride = (these_opins2.len() + max_threads - 1) / max_threads;
  let these_nets : Vec<usize> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
   let start = thread_id * stride; let end = ((thread_id + 1) * stride).min(these_opins2.len());
   (start..end).map(|i| db.pin2net[these_opins2[i]]).collect::<Vec<usize>>()
  }).collect();
  let mut these_number_of_pins : Vec<usize> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
   let start = thread_id * stride; let end = ((thread_id + 1) * stride).min(these_nets.len());
   (start..end).map(|i| db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(
    |&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).map( |&pin|  db.cell2noutputs[db.pin2cell[pin]] ).sum()
   ).collect::<Vec<usize>>()
  }).collect();
  these_number_of_pins = these_number_of_pins.into_iter().scan(0, |acc, x : usize| { *acc +=x; Some(*acc) }).collect();
  these_number_of_pins.splice(0..0, [0]);
  let these_start = Vec::from(these_number_of_pins);

  let mut items: UVec<usize> = UVec::new_zeroed(these_start[these_start.len()-1], Device::CPU);
  let mut values: UVec<u8> = UVec::new_zeroed(these_start[these_start.len()-1], Device::CPU);
  let max_opin = these_opins2.iter().max().unwrap_or(&0);
  let mut pinid2gatspiid = UVec::new_filled(max_opin+1, max_opin+1, Device::CPU);

  let num_of_gatspi_cells = these_opins2.len();

  let pinid2gatspiid_ptr = AtomicPtr::new(pinid2gatspiid.as_mut_ptr());
  (0..num_of_gatspi_cells).into_par_iter().for_each(|i| {
   unsafe {
    let ptr = pinid2gatspiid_ptr.load(Ordering::Relaxed);
    *ptr.add(these_opins2[i]) = i;
   }
  });

  let items_ptr = AtomicPtr::new(items.as_mut_ptr());
  let values_ptr = AtomicPtr::new(values.as_mut_ptr());
  (0..these_nets.len()).into_par_iter().for_each(|i| {
   let temp_items : Vec<usize> = db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(|&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).map(|&pin| db.pin2cell[pin]).
    map(|cell| Vec::from_iter((&db.cell2pin.items[db.cell2pin.start[cell]..db.cell2pin.start[cell+1]]).into_iter().filter(|&index| db.pindirect[*index] == Direction::O).map(|&index| index)) ).concat();
   let temp_edgetypes: Vec<u8>  = db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(|&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).
    map(|&pin| iter::repeat_n(stdlib_info.get_pintype(&db.celltypes[db.pin2cell[pin]], &db.pinnames[pin].1), db.cell2noutputs[db.pin2cell[pin]]).collect::<Vec<_>>() )
    .concat();
   unsafe {
    let items_ptr = items_ptr.load(Ordering::Relaxed);
    let values_ptr = values_ptr.load(Ordering::Relaxed);
    for j in 0..temp_items.len() {
     *items_ptr.add(these_start[i] + j) = pinid2gatspiid[temp_items[j]];
     *values_ptr.add(these_start[i] + j) = temp_edgetypes[j];
    }
   }
  });
  
 let these_celltypes: Vec<u16> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
  let start = thread_id * stride;
  let end = ((thread_id + 1) * stride).min(these_opins2.len());
  (start..end).map(|i| stdlib_info.get_celltype( &db.celltypes[db.pin2cell[these_opins2[i]]], &db.pinnames[these_opins2[i]].1, &db.name )).collect::<Vec<u16>>()
 }).collect();
 
 let translation_dict: Vec<(String,String)> = (0..max_threads).into_par_iter().flat_map(|thread_id| {
  let start = legit_top_ports.len() + thread_id * stride;
  let end = if thread_id == max_threads - 1 { these_celltypes.len() } else { (legit_top_ports.len() + (thread_id + 1) * stride).min(these_celltypes.len()) };
  (start..end).map(|id| ( format!("{:?}/{}", db.pinnames[these_opins2[id]].0, db.pinnames[these_opins2[id]].1).replace("HierName()/","").replace("HierName(","").replace(")/", "/") , 
   if let None = db.netnames[db.pin2net[these_opins2[id]]].2 { format!("{:?}/{}", db.netnames[db.pin2net[these_opins2[id]]].0, db.netnames[db.pin2net[these_opins2[id]]].1).replace("HierName()/","").replace("HierName(","").replace(")/", "/")  } 
   else { format!("{:?}/{}[{}]", db.netnames[db.pin2net[these_opins2[id]]].0, db.netnames[db.pin2net[these_opins2[id]]].1, db.netnames[db.pin2net[these_opins2[id]]].2.unwrap()).replace("HierName()/","").replace("HierName(","").replace(")/", "/") }
  ) ).collect::<Vec<_>>()
 }).collect();

 let mut port_dict = HashMap::new();
 for i in legit_top_ports.clone() {
  let hashkey = 
   if db.pindirect[i] == Direction::O { pinid2gatspiid[i] } else {
    db.net2pin.items[db.net2pin.start[db.pin2net[i]]..db.net2pin.start[db.pin2net[i]+1]].into_iter().filter(|&pin| db.pindirect[*pin] == Direction::O)
     .map(|&pin| pinid2gatspiid[pin]).collect::<Vec<_>>()[0]
   }; 
  let hashvalue = 
   if db.pinnames[i].2 == None { String::from(db.pinnames[i].1.clone()) } else {String::from(format!("{}[{}]", db.pinnames[i].1, db.pinnames[i].2.unwrap())) };
  port_dict.insert( hashkey , hashvalue );
 }

  Self { 
   gatspi_cells: these_opins2.clone(), gatspi_celltypes: these_celltypes, start: these_start, items: Vec::from(items), values: Vec::from(values), pinid2gatspiid: Vec::from(pinid2gatspiid), 
   num_of_gatspi_cells: these_opins2.len(), num_of_top_ports: legit_top_ports.len(), gatspi_cellname_index: translation_dict, gatspi_port_index: port_dict,
  }
 }
}
