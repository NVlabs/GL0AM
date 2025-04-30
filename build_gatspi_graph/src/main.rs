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
    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 3 || args.len() == 4,
            "Usage: {} <verilog_path> <pickle_dump_file_path> [<top_module>]", args[0]);

    let db = NetlistDB::from_sverilog_file(
        &args[1],
        args.get(3).map(|x| x.as_ref()),
        &build_gatspi_graph::StdCellPinDefs()
    ).expect("Error parsing the verilog into netlist");

    println!("Benchmark statistics for {}", args[1]);
    println!("top module: {}", db.name);
    println!("num cells:  {}", db.num_cells);
    println!("num nets:   {}", db.num_nets);
    println!("num pins:   {}", db.num_pins);

    let time_build_gatspi = clilog::stimer!("build_gatspi");
    let x = GATSPIGraph::build_graph(&db, &build_gatspi_graph::stdlib_attributes::MLCADDesignContest2025StdLib() );
    print_type(&x);
    clilog::finish!(time_build_gatspi);

    let mut ww = File::create(&args[2]).unwrap();
    serde_pickle::to_writer(&mut ww, &x, Default::default()).unwrap();
}
