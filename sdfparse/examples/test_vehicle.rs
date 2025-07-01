use std::env;
use sdfparse::*;

fn print_type<T>(_: &T) { 
 println!("{:?}", std::any::type_name::<T>());
}

fn main() {
 clilog::init_stderr_color_debug();
 clilog::enable_timer("read SDF");
 clilog::enable_timer("");
 let args: Vec<String> = env::args().collect();
 assert!(args.len() == 2 || args.len() == 3, "Usage: {} <SDF file path> [<top_module>]", args[0]);
 let time_read_sdf = clilog::stimer!("build_sdf");
 let sdfdb = sdfParse::readin_file(&args[1]);

 println!("Benchmark statistics for {}", args[1]);
 println!("DESIGN NAME: {}", sdfdb.designName);
 print_type(&sdfdb);
 clilog::finish!(time_read_sdf);

}
