use netlistdb::NetlistDB;
use std::env;
use build_gatspi_graph::*;

use std::fs::File;

fn print_type<T>(_: &T) { 
    println!("{:?}", std::any::type_name::<T>());
}


fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("netlistdb");
    clilog::enable_timer("");
    clilog::enable_timer("total_program");
    
    let time_total = clilog::stimer!("total_program");
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 3 && args.len() <= 5,
            "Usage: {} <verilog_path> <pickle_dump_file_path> [<top_module>] [<sdf_file_path>]", args[0]);

    let time_parse_verilog = clilog::stimer!("parse_verilog");
    let db = NetlistDB::from_sverilog_file(
        &args[1],
        args.get(3).map(|x| x.as_ref()),
        &build_gatspi_graph::stdlib_attributes::GL0AMGenericVlibStdCellPinDefs()
    ).expect("Error parsing the verilog into netlist");
    clilog::finish!(time_parse_verilog);

    println!("Benchmark statistics for {}", args[1]);
    println!("top module: {}", db.name);
    println!("num cells:  {}", db.num_cells);
    println!("num nets:   {}", db.num_nets);
    println!("num pins:   {}", db.num_pins);

    // Parse optional SDF file path
    let sdf_path = args.get(4).map(|path| std::path::Path::new(path));

    let time_build_gatspi = clilog::stimer!("build_gatspi");
    let x = GATSPIGraph::build_graph(&db, &build_gatspi_graph::stdlib_attributes::GL0AMStdLib(), sdf_path);
    print_type(&x);
    clilog::finish!(time_build_gatspi);

    let time_serialize = clilog::stimer!("serialize_pickle");
    let mut ww = File::create(&args[2]).unwrap();
    serde_pickle::to_writer(&mut ww, &x, Default::default()).unwrap();
    clilog::finish!(time_serialize);
    clilog::finish!(time_total);
    
}
