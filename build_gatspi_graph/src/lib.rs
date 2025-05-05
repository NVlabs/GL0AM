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
  let num_ports = db.cell2pin.start[1]; let mut ret_opins = None; let mut ret_start = None; let mut ret_items = None; let mut ret_celltypes = None;
  let mut ret_pinid2gatspiid = None;  let mut ret_num_of_gatspi_cells = None; let mut ret_values = None; let mut ret_translation_dict = None; 
  let mut ret_num_of_legit_ports = None; let mut ret_port_dict = None;
  rayon::scope(|s| {
   s.spawn(|_| {
    let mut these_opins = Vec::from_iter((num_ports..db.num_pins).filter(|&index| db.pindirect[index] == Direction::O));
    let legit_top_ports = Vec::from_iter( (0..num_ports).filter( |pin| (Some(db.pin2net[*pin]) != db.net_zero) && (Some(db.pin2net[*pin]) != db.net_one) ) );    
    these_opins.splice(0..0, legit_top_ports.clone()); 
    let these_opins2 = Vec::from(these_opins);
    let these_nets : Vec<usize> = these_opins2.clone().into_iter().map(|pin| db.pin2net[pin]).collect();
    let mut these_number_of_pins : Vec<usize> = these_nets.clone().into_iter().map(|net| db.net2pin.items[db.net2pin.start[net]..db.net2pin.start[net+1]].into_iter().filter(
                                         |&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).map( |&pin|  db.cell2noutputs[db.pin2cell[pin]] ).sum()
                                        ).scan(0, |acc, x : usize| { *acc +=x; Some(*acc) }).collect();
    these_number_of_pins.splice(0..0, [0]);
    let these_start = Vec::from(these_number_of_pins);

    let mut items = UVec::new_zeroed(these_start[these_start.len()-1], Device::CPU);
    let mut values = UVec::new_zeroed(these_start[these_start.len()-1], Device::CPU);
    let max_opin = these_opins2.iter().max().unwrap_or(&0);
    let mut pinid2gatspiid = UVec::new_filled(max_opin+1, max_opin+1, Device::CPU);
  
    let num_of_gatspi_cells = these_opins2.len();
    
    for i in 0..num_of_gatspi_cells {
     pinid2gatspiid[these_opins2[i]] = i;
    }
  
    for i in 0..these_nets.len() {
     let temp_items : &Vec<&usize> = &db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(|&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).map(|&pin| db.pin2cell[pin]).
      map(|cell| Vec::from_iter((&db.cell2pin.items[db.cell2pin.start[cell]..db.cell2pin.start[cell+1]]).into_iter().filter(|&index| db.pindirect[*index] == Direction::O)) ).concat();
     let temp_edgetypes: &_  = &db.net2pin.items[db.net2pin.start[these_nets[i]]..db.net2pin.start[these_nets[i]+1]].into_iter().filter(|&pin| (db.pindirect[*pin] == Direction::I) && (db.pin2cell[*pin] != 0) ).
      map(|&pin| iter::repeat_n(MLCADDesignContest2025StdLib().get_pintype(&db.celltypes[db.pin2cell[pin]], &db.pinnames[pin].1), db.cell2noutputs[db.pin2cell[pin]]).collect::<Vec<_>>() )
      .concat();
     for j in 0..temp_items.len() {
      items[these_start[i] + j] = pinid2gatspiid[*temp_items[j]];
      values[these_start[i] + j] = temp_edgetypes[j];
     }
    }
    
   let these_celltypes = these_opins2.clone().into_iter().map(|opin| stdlib_info.get_celltype( &db.celltypes[db.pin2cell[opin]], &db.pinnames[opin].1, &db.name )).collect::<Vec<_>>();
   
   let translation_dict : Vec<(String,String)> =  (legit_top_ports.len()..these_celltypes.len()).map(|id| ( format!("{}/{}", db.pinnames[these_opins2[id]].0.cur, db.pinnames[these_opins2[id]].1) , 
    if let None = db.netnames[db.pin2net[these_opins2[id]]].2 { String::from(db.netnames[db.pin2net[these_opins2[id]]].1.clone()) } else { format!("{}[{}]", db.netnames[db.pin2net[these_opins2[id]]].1, db.netnames[db.pin2net[these_opins2[id]]].2.unwrap()) }
    ) ).collect::<Vec<_>>();
   
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

    ret_start = Some(these_start); ret_opins = Some(these_opins2); ret_items = Some(items); ret_values = Some(values);
    ret_pinid2gatspiid = Some(pinid2gatspiid); ret_num_of_gatspi_cells = Some(num_of_gatspi_cells);
    ret_celltypes = Some(these_celltypes); ret_translation_dict = Some(translation_dict); ret_num_of_legit_ports = Some(legit_top_ports.len());
    ret_port_dict = Some(port_dict);
   });
  });

  let start = ret_start.unwrap(); let top_opins = ret_opins.unwrap(); let items = ret_items.unwrap(); let values = ret_values.unwrap();
  let pinid2gatspiid =  ret_pinid2gatspiid.unwrap(); let num_of_gatspi_cells = ret_num_of_gatspi_cells.unwrap();
  let celltypes = ret_celltypes.unwrap();   let gatspi_cellname_index = ret_translation_dict.unwrap(); let num_of_top_ports = ret_num_of_legit_ports.unwrap();
  let gatspi_port_index = ret_port_dict.unwrap();

  

  Self { 
   gatspi_cells: top_opins, gatspi_celltypes: celltypes, start: start, items: Vec::from(items), values: Vec::from(values), pinid2gatspiid: Vec::from(pinid2gatspiid), 
   num_of_gatspi_cells: num_of_gatspi_cells, num_of_top_ports: num_of_top_ports, gatspi_cellname_index: gatspi_cellname_index, gatspi_port_index: gatspi_port_index,
  }
 }
}
