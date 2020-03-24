extern crate autodiff;

use autodiff::{IScalar, ScalarType, Scalar};
use autodiff::ScalarType::{Variable, Constant};

pub fn main(){
    let x = IScalar{id: 0, kind: Variable("x".to_owned())};
    let y = IScalar{id: 1, kind: Variable("y".to_owned())};
    let mut expr = IScalar{id: 2, kind: ScalarType::Mul(Box::new(IScalar{id: 3, kind: Constant(3.)}),
                                                        Box::new(x.clone()))};
    for i in 0..2{
        expr = IScalar{id:i+4, kind: ScalarType::Mul(Box::new(x.clone()), Box::new(expr))};
    }
    expr = IScalar{id:10000, kind: ScalarType::Mul(Box::new(y.clone()), Box::new(expr))};
    println!("Expr: {}",expr.kind);
    let ds = vec![Scalar(0), Scalar(1)];
    let mut diff_graphs = expr.diff_graph(&ds);
    let diff_graph_x = diff_graphs.remove(0).optimize();
    let diff_graph_y = diff_graphs.remove(0).optimize();
    println!("Dx: {}",diff_graph_x.kind);
    println!("Dy: {}",diff_graph_y.kind);
}