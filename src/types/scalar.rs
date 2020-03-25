use crate::inflate_map;
use crate::types::id::Id;
use crate::types::pool::Pool;
use crate::types::scalar::ScalarType::*;
use std::collections::HashMap;

//#[derive(Clone)]
pub struct IScalar {
    pub(super) id: Id,
    pub(super) kind: ScalarType,
    pub(super) cache: Option<Vec<f64>>,
}

#[derive(Clone, Copy)]
pub enum ScalarType {
    Constant(f64),
    Variable,
    Add(Id, Id),
    Mul(Id, Id),
}

impl ScalarType {
    pub fn is_zero(&self) -> bool {
        match self {
            Constant(val) => *val == 0.,
            _ => false,
        }
    }
    pub fn is_one(&self) -> bool {
        match self {
            Constant(val) => *val == 1.,
            _ => false,
        }
    }
}
impl PartialEq for IScalar {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Pool {
    pub fn to_string(&self, id: Id) -> String {
        self.get(id).to_string(self)
    }
    //Calculates a single scalar node given bound variables; batch size 1
    //Bound variables can either be provided via function argument or via update_bound_variables
    pub fn evaluate_scalar(&mut self, id: Id, bound_variables: &HashMap<Id, f64>) -> f64 {
        self.update_bound_variables(&inflate_map(bound_variables));
        self.ievaluate(id, self.env.batch_size()).pop().unwrap()
    }

    //Calculates a single scalar node given bound variables over a batch; batch size variable
    pub fn evaluate_scalar_batch(
        &mut self,
        id: Id,
        bound_variables: &HashMap<Id, Vec<f64>>,
    ) -> Vec<f64> {
        self.update_bound_variables(bound_variables);
        self.ievaluate(id, self.env.batch_size())
    }

    //Calculates multiple scalar nodes given bound variables; batch size 1
    pub fn evaluate_scalars(
        &mut self,
        id: Vec<Id>,
        bound_variables: &HashMap<Id, f64>,
    ) -> Vec<f64> {
        self.update_bound_variables(&inflate_map(bound_variables));
        id.iter()
            .map(|i| self.ievaluate(*i, self.env.batch_size()).pop().unwrap())
            .collect()
    }

    //Calculates multpile scalar nodes given bound variabes over a batch; batch size variable
    //Return type: Vec<Vec<f64>>, First Dimension: Node, Second Dimension: Batch
    pub fn evaluate_scalars_batch(
        &mut self,
        id: Vec<Id>,
        bound_variables: &HashMap<Id, Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        self.update_bound_variables(bound_variables);
        id.iter()
            .map(|i| self.ievaluate(*i, self.env.batch_size()))
            .collect()
    }

    fn ievaluate(&mut self, id: Id, batch_size: usize) -> Vec<f64> {
        let mut scalar = self.calculate(id);
        let ret = if let Some(res) = &scalar.cache {
            res.clone()
        } else {
            let res = match scalar.kind {
                Constant(val) => vec![val; batch_size],
                Add(bval1, bval2) => {
                    let bval1 = self.ievaluate(bval1, batch_size);
                    let bval2 = self.ievaluate(bval2, batch_size);
                    bval1.iter().zip(bval2.iter()).map(|(a, b)| a + b).collect()
                }
                Mul(bval1, bval2) => {
                    let bval1 = self.ievaluate(bval1, batch_size);
                    let bval2 = self.ievaluate(bval2, batch_size);
                    bval1.iter().zip(bval2.iter()).map(|(a, b)| a * b).collect()
                }
                Variable => self.env.get(id, self),
            };
            scalar.cache = Some(res.clone());
            res
        };
        self.finished_calculating(scalar);
        ret
    }

    pub fn diff_graph(&mut self, id: Id, ds: &Vec<Id>) -> Vec<Id> {
        let scalar = self.get(id);
        match scalar.kind {
            ScalarType::Constant(_) => (0..ds.len())
                .map(|_| self.register_scalar_constant(0.))
                .collect(),
            ScalarType::Variable => ds
                .iter()
                .map(|dx| {
                    if *dx == id {
                        self.register_scalar_constant(1.)
                    } else {
                        self.register_scalar_constant(0.)
                    }
                })
                .collect(),
            ScalarType::Add(bl, br) => {
                let nbl = self.diff_graph(bl, ds);
                let nbr = self.diff_graph(br, ds);
                let res : Vec<Id>= nbl.into_iter()
                    .zip(nbr.into_iter())
                    .map(|(a, b)| self.add(a, b))
                    .collect();
                self.optimize_multiple_once(&res);
                res
            }
            ScalarType::Mul(bl, br) => {
                let nbl = self.diff_graph(bl, ds);
                let nbr = self.diff_graph(br, ds);
                let res: Vec<Id> = nbl.into_iter()
                    .zip(nbr.into_iter())
                    .map(|(a, b)| {
                        let l = self.mul(a, br);
                        let r = self.mul(bl, b);
                        self.optimize_multiple_once(&[l,r]);
                        self.add(l, r)
                    })
                    .collect();
                self.optimize_multiple_once(&res);
                res
            }
        }
    }

    //Optimizing IScalar's can make certain Scalars be reference dead
    pub fn optimize(&mut self, id: Id) {
        let mut scalar = self.calculate(id);
        match scalar.kind {
            ScalarType::Variable | ScalarType::Constant(_) => self.optimize_once(&mut scalar),
            ScalarType::Add(bl, br) | ScalarType::Mul(bl, br) => {
                self.optimize(bl);
                self.optimize(br);
                self.optimize_once(&mut scalar);
            }
        }
        self.finished_calculating(scalar);
    }

    pub(crate) fn optimize_once(&mut self, scalar: &mut IScalar) {
        match scalar.kind {
            ScalarType::Variable | ScalarType::Constant(_) => {}
            ScalarType::Add(bl, br) => {
                let nbl = self.get(bl);
                let nbr = self.get(br);
                if nbl.is_zero() && nbr.is_zero() {
                    scalar.kind = ScalarType::Constant(0.);
                } else if nbl.is_zero() {
                    scalar.kind = nbr.kind;
                } else if nbr.is_zero() {
                    scalar.kind = nbl.kind
                }
                if bl == br {
                    scalar.kind = Mul(self.register_scalar_constant(2.), bl);
                }
            }
            ScalarType::Mul(bl, br) => {
                let nbl = self.get(bl);
                let nbr = self.get(br);
                if nbl.is_zero() || nbr.is_zero() {
                    scalar.kind = ScalarType::Constant(0.);
                } else if nbl.is_one() {
                    scalar.kind = nbr.kind;
                } else if nbr.is_one() {
                    scalar.kind = nbl.kind;
                }
            }
        }
    }

    pub fn optimize_multiple(&mut self, ids: &[Id]) {
        ids.iter().for_each(|id| self.optimize(*id))
    }
    pub fn optimize_multiple_once(&mut self, ids: &[Id]) {
        ids.iter().for_each(|id| {
            let mut scalar = self.calculate(*id);
            self.optimize_once(&mut scalar);
            self.finished_calculating(scalar);
        })
    }
}
impl<'a> IScalar {
    pub fn is_zero(&self) -> bool {
        self.kind.is_zero()
    }
    pub fn is_one(&self) -> bool {
        self.kind.is_one()
    }
    pub fn to_string(&self, pool: &Pool) -> String {
        self.i_to_string(pool, 0)
    }
    fn i_to_string(&self, pool: &Pool, prio: usize) -> String {
        match &self.kind {
            Constant(val) => format!("{}", val),
            Variable => pool
                .find_variable_name(self.id)
                .expect("Variable in graph but not in namepsace"),
            Add(bl, br) => {
                let (l, r) = (
                    pool.get(*bl).i_to_string(pool, 0),
                    pool.get(*br).i_to_string(pool, 0),
                );
                if prio == 1 {
                    format!("({}+{})", l, r)
                } else {
                    format!("{}+{}", l, r)
                }
            }
            Mul(bl, br) => {
                let (l, r) = (
                    pool.get(*bl).i_to_string(pool, 1),
                    pool.get(*br).i_to_string(pool, 1),
                );
                format!("{}*{}", l, r)
            }
        }
    }
}
