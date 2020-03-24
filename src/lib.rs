use std::collections::HashMap;
use std::fmt::{Display, Error, Formatter};

#[derive(Copy, Clone)]
pub struct Scalar(pub usize);

#[derive(Clone)]
pub struct IScalar {
    pub id: usize,
    pub kind: ScalarType,
}

#[derive(Clone)]
pub enum ScalarType {
    Constant(f64),
    Variable(String),
    Add(Box<IScalar>, Box<IScalar>),
    Mul(Box<IScalar>, Box<IScalar>),
}

impl ScalarType {
    pub fn is_zero(&self) -> bool {
        match self {
            ScalarType::Constant(val) => *val == 0.,
            _ => false,
        }
    }
    pub fn is_one(&self) -> bool {
        match self {
            ScalarType::Constant(val) => *val == 1.,
            _ => false,
        }
    }
    fn to_string(&self, prio: usize) -> String {
        match self {
            ScalarType::Constant(val) => format!("{}", val),
            ScalarType::Variable(s) => s.clone(),
            ScalarType::Add(bl, br) => {
                let (l, r) = (bl.kind.to_string(0), br.kind.to_string(0));
                if prio == 1 {
                    format!("({}+{})", l, r)
                } else {
                    format!("{}+{}", l, r)
                }
            }
            ScalarType::Mul(bl, br) => {
                let (l, r) = (bl.kind.to_string(1), br.kind.to_string(1));
                format!("{}*{}", l, r)
            }
        }
    }
}
impl Display for ScalarType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string(0))
    }
}
impl PartialEq for IScalar {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<'a> IScalar {
    pub fn is_zero(&self) -> bool {
        self.kind.is_zero()
    }
    pub fn is_one(&self) -> bool {
        self.kind.is_one()
    }
    pub fn evaluate(&self, bound_variables: &HashMap<usize, f64>) -> f64 {
        let x: HashMap<usize, Vec<f64>> = bound_variables
            .iter()
            .map(|(k, v)| (*k, vec![*v]))
            .collect();
        self.evaluate_batch(&x, 1).pop().unwrap()
    }

    pub fn evaluate_batch(
        &self,
        bound_variables: &HashMap<usize, Vec<f64>>,
        batch_size: usize,
    ) -> Vec<f64> {
        match &self.kind {
            ScalarType::Constant(val) => vec![*val; batch_size],
            ScalarType::Add(bval1, bval2) => {
                let bval1 = (*bval1).evaluate_batch(&bound_variables, batch_size);
                let bval2 = (*bval2).evaluate_batch(&bound_variables, batch_size);
                bval1.iter().zip(bval2.iter()).map(|(a, b)| a + b).collect()
            }
            ScalarType::Mul(bval1, bval2) => {
                let bval1 = (*bval1).evaluate_batch(&bound_variables, batch_size);
                let bval2 = (*bval2).evaluate_batch(&bound_variables, batch_size);
                bval1.iter().zip(bval2.iter()).map(|(a, b)| a * b).collect()
            }
            ScalarType::Variable(_) => {
                if let Some(val) = bound_variables.get(&self.id) {
                    val.clone()
                } else {
                    panic!(format!(
                        "Unbound variable with id {} wasn't provided a value!",
                        self.id
                    ))
                }
            }
        }
    }

    pub fn diff_graph(&self, ds: &Vec<Scalar>) -> Vec<IScalar> {
        match &self.kind {
            ScalarType::Constant(_) => (0..ds.len())
                .map(|_| IScalar {
                    id: 0,
                    kind: ScalarType::Constant(0.),
                })
                .collect(),
            ScalarType::Variable(_) => ds
                .iter()
                .map(|dx| {
                    if dx.0 == self.id {
                        IScalar {
                            id: 0,
                            kind: ScalarType::Constant(1.),
                        }
                    } else {
                        IScalar {
                            id: 0,
                            kind: ScalarType::Constant(0.),
                        }
                    }
                })
                .collect(),
            ScalarType::Add(bl, br) => {
                let nbl = (*bl).diff_graph(ds);
                let nbr = (*br).diff_graph(ds);
                nbl.into_iter()
                    .zip(nbr.into_iter())
                    .map(|(a, b)| IScalar {
                        id: 0,
                        kind: ScalarType::Add(Box::new(a), Box::new(b)),
                    })
                    .collect()
            }
            ScalarType::Mul(bl, br) => {
                let nbl = (*bl).diff_graph(ds);
                let nbr = (*br).diff_graph(ds);
                nbl.into_iter()
                    .zip(nbr.into_iter())
                    .map(|(a, b)| IScalar {
                        id: 0,
                        kind: ScalarType::Add(
                            Box::new(IScalar {
                                id: 0,
                                kind: ScalarType::Mul(Box::new(a), Box::new(*br.clone())),
                            }),
                            Box::new(IScalar {
                                id: 0,
                                kind: ScalarType::Mul(Box::new(*bl.clone()), Box::new(b)),
                            }),
                        ),
                    })
                    .collect()
            }
        }
    }

    //Optimizing IScalar's can remove certain IScalars from the graph
    pub fn optimize(self) -> Self {
        match self.kind {
            ScalarType::Variable(_) | ScalarType::Constant(_) => self,
            ScalarType::Add(bl, br) => {
                let nbl = (*bl).optimize();
                let nbr = (*br).optimize();
                if nbl.is_zero() && nbr.is_zero() {
                    IScalar {
                        id: self.id,
                        kind: ScalarType::Constant(0.),
                    }
                } else if nbl.is_zero() {
                    nbr
                } else if nbr.is_zero() {
                    nbl
                } else {
                    IScalar {
                        id: self.id,
                        kind: ScalarType::Add(Box::new(nbl), Box::new(nbr)),
                    }
                }
            }
            ScalarType::Mul(bl, br) => {
                let nbl = (*bl).optimize();
                let nbr = (*br).optimize();
                if nbl.is_zero() || nbr.is_zero() {
                    IScalar {
                        id: self.id,
                        kind: ScalarType::Constant(0.),
                    }
                } else if nbl.is_one() {
                    nbr
                } else if nbr.is_one() {
                    nbl
                } else {
                    IScalar {
                        id: self.id,
                        kind: ScalarType::Mul(Box::new(nbl), Box::new(nbr)),
                    }
                }
            }
        }
    }
}
