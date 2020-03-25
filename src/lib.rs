use crate::types::id::Id;
use std::collections::HashMap;

pub mod types;
pub(crate) fn inflate_map(map: &HashMap<Id, f64>) -> HashMap<Id, Vec<f64>> {
    map.iter().map(|(k, v)| (*k, vec![*v])).collect()
}
