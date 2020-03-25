extern crate autodiff;

use autodiff::types::pool::Pool;
use std::collections::HashMap;

pub fn main(){
    let mut pool = Pool::default();
    let x = pool.register_scalar_variable("x".to_owned());
    let y = pool.register_scalar_variable("y".to_owned());
    let three = pool.register_scalar_constant(3.);
    let mut expr = pool.mul(three, x);
    for _ in 0..2{
        expr = pool.mul(x, expr);
    }
    expr = pool.mul(expr, y);
    println!("Expr: {}",pool.to_string(expr));
    let ds = vec![x,y];
    let mut diff_graphs = pool.diff_graph(expr, &ds);
    //let diff_graph_x = diff_graphs.remove(0).optimize();
    //let diff_graph_y = diff_graphs.remove(0).optimize();
    println!("Dx: {}",pool.to_string(diff_graphs[0]));
    println!("Dy: {}",pool.to_string(diff_graphs[1]));
    let mut map =HashMap::new();
    map.insert(x, vec![-1.,0.,1.]);
    map.insert(y, vec![0.,0.,1.]);
    let (dxdy) = pool.evaluate_scalars_batch(diff_graphs, &map);
    println!("Dx, Dy at [-1,0,1]: {:?}",dxdy);
    pool.print_state();
}