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

#[derive(Clone)]
pub enum ScalarType {
    Constant(f64),
    Variable(Id),
    Add(Vec<Id>),
    Mul(Vec<Id>),
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
    pub fn is_constant(&self) -> bool {
        match self {
            Constant(_) => true,
            _ => false,
        }
    }
    pub fn unwrap_constant(&self) -> f64 {
        match self {
            Constant(val) => *val,
            _ => panic!("Trying to unwrap constant on a non constant!"),
        }
    }
    pub fn is_mul(&self) -> bool {
        match self {
            Mul(_) => true,
            _ => false,
        }
    }
    pub fn unwrap_mul(&self) -> Vec<Id> {
        match self {
            Mul(ids) => ids.clone(),
            _ => panic!("Trying to unwra mul on a non mul!"),
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
            let res = match &scalar.kind {
                Constant(val) => vec![*val; batch_size],
                Add(ids) => {
                    let vals: Vec<Vec<f64>> = ids
                        .iter()
                        .map(|id| self.ievaluate(*id, batch_size))
                        .collect();
                    vals.iter().fold(vec![0.; batch_size], |curr, next| {
                        curr.iter().zip(next.iter()).map(|(a, b)| *a + *b).collect()
                    })
                }
                Mul(ids) => {
                    if ids.len() == 0 {
                        vec![0; batch_size];
                    }
                    let vals: Vec<Vec<f64>> = ids
                        .iter()
                        .map(|id| self.ievaluate(*id, batch_size))
                        .collect();
                    vals.iter().fold(vec![1.; batch_size], |curr, next| {
                        curr.iter().zip(next.iter()).map(|(a, b)| *a * *b).collect()
                    })
                }
                Variable(id) => self.env.get(*id, self),
            };
            scalar.cache = Some(res.clone());
            res
        };
        self.finished_calculating(scalar);
        ret
    }

    pub fn diff_graph(&mut self, id: Id, ds: &Vec<Id>) -> Vec<Id> {
        let scalar = self.calculate(id);
        let ret = match &scalar.kind {
            ScalarType::Constant(_) => (0..ds.len())
                .map(|_| self.register_scalar_constant(0.))
                .collect(),
            ScalarType::Variable(nid) => ds
                .iter()
                .map(|dx| {
                    if *dx == *nid {
                        self.register_scalar_constant(1.)
                    } else {
                        self.register_scalar_constant(0.)
                    }
                })
                .collect(),
            ScalarType::Add(ids) => {
                let graphs: Vec<Vec<Id>> = ids.iter().map(|id| self.diff_graph(*id, ds)).collect();
                let res = graphs.iter().fold(vec![vec![]; ids.len()], |curr, next| {
                    curr.into_iter()
                        .zip(next.iter())
                        .map(|(mut a, b)| {
                            a.push(*b);
                            a
                        })
                        .collect()
                });
                let res: Vec<Id> = res.into_iter().map(|ids| self.add(ids)).collect();
                self.optimize_multiple_once(&res);
                res
            }
            ScalarType::Mul(ids) => {
                //Split in half
                debug_assert!(ids.len() > 0);
                if ids.len() == 1 {
                    self.diff_graph(ids[0], ds)
                } else {
                    let mid = ids.len() / 2;
                    let left = ids[0..mid].to_vec();
                    let left = self.mul(left);
                    let right = ids[mid..].to_vec();
                    let right = self.mul(right);
                    self.optimize_multiple_once(&[left, right]);
                    let dleft = self.diff_graph(left, ds);
                    let dright: Vec<Id> = self.diff_graph(right, ds);
                    let res: Vec<Id> = dleft
                        .into_iter()
                        .zip(dright.into_iter())
                        .map(|(a, b)| {
                            let l = self.mul(vec![a, right]);
                            let r = self.mul(vec![left, b]);
                            self.optimize_multiple_once(&[l, r]);
                            self.add(vec![l, r])
                        })
                        .collect();
                    self.optimize_multiple_once(&res);
                    res
                }
            }
        };
        self.finished_calculating(scalar);
        ret
    }

    //Optimizing IScalar's can make certain Scalars be reference dead
    /* pub fn optimize(&mut self, id: Id) {
        let mut scalar = self.calculate(id);
        match &scalar.kind {
            ScalarType::Variable(_) | ScalarType::Constant(_) => self.optimize_once(&mut scalar),
            ScalarType::Add(ids) | ScalarType::Mul(ids) => {
                ids.iter().for_each(|id| self.optimize(*id));
                self.optimize_once(&mut scalar);
            }
        }
        self.finished_calculating(scalar);
    }*/

    pub(crate) fn optimize_once(&mut self, scalar: &mut IScalar) {
        match scalar.kind.clone() {
            ScalarType::Variable(_) | ScalarType::Constant(_) => {}
            ScalarType::Add(mut ids) => {
                debug_assert!(ids.len() > 1);
                //Add all Adds from childs
                let mut append = Vec::new();
                let mut i = 0;
                while i < ids.len() {
                    let other = self.get(ids[i]);
                    if let Add(other_ids) = &other.kind {
                        other_ids.iter().for_each(|id| append.push(*id));
                        ids.remove(i);
                    } else {
                        i += 1;
                    }
                }
                ids.append(&mut append);
                //Remove constants
                let constant_sum = ids.iter().fold(0., |curr, next| {
                    let scalar = self.get(*next);
                    if scalar.is_constant() {
                        curr + scalar.unwrap_constant()
                    } else {
                        curr
                    }
                });
                ids.retain(|elem| {
                    let scalar = self.get(*elem);
                    !scalar.is_constant()
                });
                if constant_sum != 0. {
                    let constant = self.register_scalar_constant(constant_sum);
                    ids.push(constant)
                }
                if ids.len() == 0 {
                    scalar.kind = ScalarType::Constant(0.);
                } else {
                    //Adds zusammenfassen von gleichen Id's
                    ids.sort_by(|a, b| {
                        if a.0 > b.0 {
                            std::cmp::Ordering::Less
                        } else if a.0 < b.0 {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    });
                    i = 1;
                    let mut equal_count = 1;
                    while i < ids.len() {
                        if ids[i] == ids[i - 1] {
                            ids.remove(i);
                            equal_count += 1;
                        } else {
                            if equal_count > 1 {
                                let other = ids.remove(i - 1);
                                let times = self.register_scalar_constant(equal_count as f64);
                                let mul = self.mul(vec![times, other]);
                                ids.push(mul);
                            }
                            equal_count = 1;
                            i += 1;
                        }
                    }

                    if ids.len() == 1 {
                        scalar.kind = self.get(ids[0]).kind.clone();
                    } else {
                        scalar.kind = Add(ids);
                    }
                }
            }
            ScalarType::Mul(mut ids) => {
                debug_assert!(ids.len() > 1);
                //Add all muls from childs
                let mut append = Vec::new();
                let mut i = 0;
                while i < ids.len() {
                    let other = self.get(ids[i]);
                    if let Mul(other_ids) = &other.kind {
                        other_ids.iter().for_each(|id| append.push(*id));
                        ids.remove(i);
                    } else {
                        i += 1;
                    }
                }
                ids.append(&mut append);
                //Remove constants
                let constant_sum = ids.iter().fold(1., |curr, next| {
                    let scalar = self.get(*next);
                    if scalar.is_constant() {
                        curr * scalar.unwrap_constant()
                    } else {
                        curr
                    }
                });
                ids.retain(|elem| {
                    let scalar = self.get(*elem);
                    !scalar.is_constant()
                });
                if constant_sum == 0. {
                    scalar.kind = ScalarType::Constant(0.);
                } else {
                    if constant_sum != 1. {
                        ids.push(self.register_scalar_constant(constant_sum));
                    }
                    if ids.len() == 0 {
                        scalar.kind = ScalarType::Constant(1.);
                    } else {
                        ids.sort_by(|a, b| {
                            if a.0 > b.0 {
                                std::cmp::Ordering::Less
                            } else if a.0 < b.0 {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        });
                        //Muls zusammenfassen von gleichen Id's
                        if ids.len() == 1 {
                            scalar.kind = self.get(ids[0]).kind.clone();
                        } else {
                            scalar.kind = Mul(ids);
                        }
                    }
                }
            }
        }
    }

    fn distributive_optimizations(&mut self, ids: &Vec<Id>) {
        let mult_ids: Vec<Id> = ids
            .clone()
            .into_iter()
            .filter(|id| {
                let scalar = self.get(*id);
                scalar.is_mul()
            })
            .collect();
        let mult_vecs: Vec<Vec<Id>> = mult_ids
            .into_iter()
            .map(|id| {
                let scalar = self.get(id);
                scalar.unwrap_mul()
            })
            .collect();
        if mult_vecs.is_empty() {
            return;
        }
        //Step 1. Find biggest shared constant
        //Filter constants:
        let constants: Vec<f64> = mult_vecs
            .clone()
            .into_iter()
            .map(|vec| {
                //Extra the constant of each of mult vec's
                let constant: Vec<Id> = vec
                    .into_iter()
                    .filter(|id| {
                        let scalar = self.get(*id);
                        scalar.is_constant()
                    })
                    .collect();
                if constant.is_empty() {
                    1.0
                } else if constant.len() == 1 {
                    self.get(constant[0]).unwrap_constant()
                } else {
                    panic!("There should only be one constant after optimizations")
                }
            })
            .collect();
        let largest_constant: f64 = constants
            .iter()
            .map(|f| f.abs())
            .max_by(|&f1, &f2| {
                if f1 < f2 {
                    std::cmp::Ordering::Less
                } else if f2 < f1 {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .unwrap();
        let positive_constants: Vec<f64> = constants.clone().into_iter().filter(|f| *f >= 0.).collect();
        let positive = positive_constants.len() >= constants.len() / 2;
        let constant = if positive {
            largest_constant
        } else {
            -largest_constant
        };
        let constant_id = self.register_scalar_constant(constant);
        //Filter out all constants
        let mult_vecs: Vec<Vec<Id>> = mult_vecs
            .into_iter()
            .map(|vec| {
                vec.into_iter()
                    .filter(|id| !self.get(*id).is_constant())
                    .collect()
            })
            .collect();
        //Step 2. Shared powers
        //Step 3. Find other shared values, by ID
        let maxmin_id = mult_vecs
            .iter()
            .map(|vec| vec[vec.len() - 1].0)
            .max_by(|&u1, &u2| {
                if u1 < u2 {
                    std::cmp::Ordering::Less
                } else if u2 < u1 {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .unwrap();
        fn get_max_id(pos: &Vec<usize>, mult_vecs: &Vec<Vec<Id>>) -> Option<(usize, usize)> {
            let mut max: Option<(usize, usize)> = None;
            for (i, p) in pos.iter().enumerate() {
                if mult_vecs.len() > *p {
                    let m = mult_vecs[i][*p];
                    if max.is_none() || max.unwrap().1 < m.0 {
                        max = Some((i, m.0));
                    }
                }
            }
            max
        }
    }
    /*pub fn optimize_multiple(&mut self, ids: &[Id]) {
        ids.iter().for_each(|id| self.optimize(*id))
    }*/
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
    pub fn is_constant(&self) -> bool {
        self.kind.is_constant()
    }
    pub fn unwrap_constant(&self) -> f64 {
        self.kind.unwrap_constant()
    }
    pub fn is_mul(&self) -> bool {
        self.kind.is_mul()
    }
    pub fn unwrap_mul(&self) -> Vec<Id> {
        self.kind.unwrap_mul()
    }
    pub fn to_string(&self, pool: &Pool) -> String {
        self.i_to_string(pool, 0)
    }
    fn i_to_string(&self, pool: &Pool, prio: usize) -> String {
        match &self.kind {
            Constant(val) => format!("{}", val),
            Variable(id) => pool
                .find_variable_name(*id)
                .expect("Variable in graph but not in namepsace"),
            Add(ids) => {
                let child_strs: Vec<String> = ids
                    .iter()
                    .map(|id| pool.get(*id).i_to_string(pool, 0))
                    .collect();
                let mut res_str = String::new();
                if prio == 1 {
                    res_str.push('(');
                }
                res_str.push_str(&child_strs.join("+"));
                if prio == 1 {
                    res_str.push(')');
                }
                res_str
            }
            Mul(ids) => {
                let child_str: Vec<String> = ids
                    .iter()
                    .map(|id| pool.get(*id).i_to_string(pool, 1))
                    .collect();
                let mut res_str = String::new();
                res_str.push_str(&child_str.join("*"));
                res_str
            }
        }
    }
}
