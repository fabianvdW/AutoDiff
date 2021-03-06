use crate::types::id::Id;
use crate::types::pool::ExpressionStatus::Expression;
use crate::types::scalar::{IScalar, ScalarType};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

pub struct Env {
    variable_mapping: HashMap<Id, Vec<f64>>,
    batch_size: Option<usize>,
}
impl Env {
    pub fn clear(&mut self) {
        self.batch_size = None;
        self.variable_mapping.clear();
    }
    pub fn insert(&mut self, id: Id, vec: Vec<f64>) {
        let len = vec.len();
        if let Some(size) = self.batch_size {
            if len != size {
                panic!("Invalid batch size")
            }
        } else {
            self.batch_size = Some(len)
        }
        self.variable_mapping.insert(id, vec);
    }

    pub fn batch_size(&self) -> usize {
        if self.batch_size.is_none() {
            1
        } else {
            self.batch_size.unwrap()
        }
    }

    pub fn get(&self, id: Id, pool: &Pool) -> Vec<f64> {
        if let Some(res) = self.variable_mapping.get(&id) {
            res.clone()
        } else {
            if let Ok(_) = pool.exists(id) {
                let var_name = pool.find_variable_name(id);
                if var_name.is_none() {
                    panic!(format!(
                        "Calling env.get on an IScalar in the graph which is not in the namespace"
                    ));
                } else {
                    panic!(format!(
                        "Unbound variable {}, id {} does not exist in env",
                        var_name.unwrap(),
                        id.0
                    ));
                }
            } else {
                panic!(format!(
                    "Unbound variable with id {} does not exist in the compute graph",
                    id.0
                ))
            }
        }
    }
}
impl Display for Env {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Mapping: {:?}\n Batch size: {:?}",
            self.variable_mapping, self.batch_size
        )
    }
}
impl Default for Env {
    fn default() -> Self {
        Env {
            variable_mapping: HashMap::new(),
            batch_size: None,
        }
    }
}
pub enum ExpressionStatus {
    Expression(IScalar),
    None,
    Calculating,
}
impl ExpressionStatus {
    pub fn is_some(&self) -> bool {
        match self {
            ExpressionStatus::Expression(_) => true,
            _ => false,
        }
    }
    pub fn to_string(&self, pool: &Pool) -> String {
        match self {
            ExpressionStatus::None => "None".to_owned(),
            ExpressionStatus::Calculating => "Calculating".to_owned(),
            ExpressionStatus::Expression(scalar) => scalar.to_string(pool),
        }
    }
}
pub struct Pool {
    id_counter: Id,
    namespace: HashMap<String, Id>,
    expr_tree: Vec<ExpressionStatus>,
    pub(crate) env: Env,
}
impl Pool {
    fn id(&mut self) -> Id {
        let res = self.id_counter;
        self.id_counter = Id(self.id_counter.0 + 1);
        res
    }

    fn invalidate_caches(&mut self) {
        self.expr_tree.iter_mut().for_each(|opt| {
            match opt {
                ExpressionStatus::Expression(scalar) => scalar.cache = None,
                ExpressionStatus::Calculating => {
                    panic!("Invalidating caches while calculating is not allowed.")
                }
                _ => {}
            }
            ()
        })
    }

    fn exists(&self, id: Id) -> Result<(), String> {
        if id.0 >= self.expr_tree.len()
            || match self.expr_tree[id.0] {
                ExpressionStatus::None => true,
                _ => false,
            }
        {
            Err(format!("Provided id {} does not exist", id.0))
        } else {
            Ok(())
        }
    }

    pub fn debug_scalar(&self, id: Id){
        let scalar = self.get(id);
        println!("Scalar with Id {} is {}", id.0, match &scalar.kind{
            ScalarType::Variable(id)=>format!("Variable {}",self.find_variable_name(*id).unwrap()),
            ScalarType::Constant(value)=>format!("Constant: {}",*value),
            ScalarType::Add(value)=>format!("Add ids: {:?}",value),
            ScalarType::Mul(value)=>format!("Mul ids: {:?}",value)
        });
        match &scalar.kind{
            ScalarType::Add(value)|ScalarType::Mul(value)=>{
                value.iter().for_each(|id|self.debug_scalar(*id));
            }
            _=>{}
        }
    }
    //This function should only be used for debugging purposes as it's runtime is bad
    pub(crate) fn find_variable_name(&self, id: Id) -> Option<String> {
        let matches: Vec<(&String, &Id)> =
            self.namespace.iter().filter(|(_, v)| **v == id).collect();
        if let Some(s) = matches.get(0) {
            Some(s.0.clone())
        } else {
            None
        }
    }

    pub(crate) fn get(&self, id: Id) -> &IScalar {
        if let Err(debug) = self.exists(id) {
            panic!(debug);
        }
        let expr = &self.expr_tree[id.0];
        match expr{
            ExpressionStatus::Calculating => panic!(format!("Trying to get a node with id {} while it is already being calculated. This is most likely due to a cycle in the graph.", id.0)),
            ExpressionStatus::None => panic!("already checked this case"),
            ExpressionStatus::Expression(scalar) => scalar
        }
    }

    pub(crate) fn check(&self, id: Id) {
        if let Err(debug) = self.exists(id) {
            panic!(debug);
        }
        if !self.expr_tree[id.0].is_some() {
            panic!(format!("Provided id {} does not exist", id.0));
        }
    }

    pub(crate) fn calculate(&mut self, id: Id) -> IScalar {
        if let Err(debug) = self.exists(id) {
            panic!(debug);
        }
        let expr = self.expr_tree.remove(id.0);
        let scalar = match expr{
            Expression(scalar) => scalar,
            ExpressionStatus::Calculating => panic!(format!("Trying to calcutate the node with id {} while it is already being calculated. This is most likely due to a cycle in the graph", id.0)),
            ExpressionStatus::None => panic!("already checked")
        };
        self.expr_tree.insert(id.0, ExpressionStatus::Calculating);
        scalar
    }

    pub(crate) fn finished_calculating(&mut self, scalar: IScalar) {
        if let Err(debug) = self.exists(scalar.id) {
            panic!(debug);
        }
        let expr = self.expr_tree.remove(scalar.id.0);
        match expr {
            ExpressionStatus::Expression(_) => panic!(format!(
                "Finished calculating the node with id {} while it was not being calculated",
                scalar.id.0
            )),
            ExpressionStatus::None => panic!("already checked"),
            _ => {}
        };
        self.expr_tree.insert(scalar.id.0, Expression(scalar));
    }

    pub fn register_scalar_variable(&mut self, name: String) -> Id {
        //Check if variable exists
        if let Some(_) = self.namespace.get(&name) {
            panic!(format!("Scalar variable {} already exists", name));
        }
        let res = self.id();
        self.namespace.insert(name.clone(), res);
        self.expr_tree.push(Expression(IScalar {
            id: res,
            kind: ScalarType::Variable(res),
            cache: None,
        }));
        res
    }

    pub fn register_scalar_constant(&mut self, value: f64) -> Id {
        let res = self.id();
        self.expr_tree.push(Expression(IScalar {
            id: res,
            kind: ScalarType::Constant(value),
            cache: None,
        }));
        res
    }

    pub fn set_scalar(&mut self, id: Id, value: f64) {
        self.invalidate_caches();
        self.env.insert(id, vec![value]);
    }

    pub fn mul(&mut self, ids: Vec<Id>) -> Id {
        ids.iter().for_each(|id|self.check(*id));
        if ids.is_empty(){
            panic!("Can't provide empty vec to mul");
        }else if ids.len() ==1{
            ids[0]
        }else {
            let mut scalar = IScalar {
                id:self.id_counter,
                kind: ScalarType::Mul(ids),
                cache: None,
            };
            self.optimize_once(&mut scalar);
            match &scalar.kind {
                ScalarType::Variable(id) => *id,
                ScalarType::Constant(_) => {
                    let id = self.id();
                    self.expr_tree.push(Expression(scalar));
                    id
                },
                ScalarType::Mul(ids)|ScalarType::Add(ids)=>{
                    if ids.len() == 1{
                        ids[0]
                    }else {
                        let id = self.id();
                        self.expr_tree.push(Expression(scalar));
                        id
                    }
                }
            }
        }
    }

    pub fn add(&mut self, ids: Vec<Id>) -> Id {
        ids.iter().for_each(|id|self.check(*id));
        if ids.is_empty(){
            panic!("Can't provide empty vec to add");
        }else if ids.len() == 1{
            ids[0]
        }else{
            let mut scalar = IScalar {
                id:self.id_counter,
                kind: ScalarType::Add(ids),
                cache: None,
            };
            self.optimize_once(&mut scalar);
            match &scalar.kind{
                ScalarType::Variable(id)=> *id,
                ScalarType::Add(ids)|ScalarType::Mul(ids)=>{
                    if ids.len() == 1{
                        ids[0]
                    }else{
                        let id =self.id();
                        self.expr_tree.push(Expression(scalar));
                        id
                    }
                }
                ScalarType::Constant(_)=>{
                    let id = self.id();
                    self.expr_tree.push(Expression(scalar));
                    id
                }
            }
        }
    }

    pub fn update_bound_variables(&mut self, other: &HashMap<Id, Vec<f64>>) {
        if other.len() > 0 {
            self.invalidate_caches();
        }
        other.iter().for_each(|(id, f)| {
            self.env.insert(*id, f.clone());
        });
    }

    pub fn clear_bound_variables(&mut self) {
        self.env.clear();
    }

    pub fn print_state(&self) {
        println!("<Pool object with:\nCurrent Id counter: {}\n\t<NameSpace with:\n\t{:?}>\n\t<Env with:\n\t{}>\nExpressions: ",self.id_counter.0, self.namespace,self.env);
        for (i, expr) in self.expr_tree.iter().enumerate() {
            println!("Id {}: \"{}\"", i, expr.to_string(self));
        }
        println!(">");
    }
}
impl Default for Pool {
    fn default() -> Self {
        Pool {
            id_counter: Id(0),
            namespace: HashMap::new(),
            expr_tree: vec![],
            env: Env::default(),
        }
    }
}
