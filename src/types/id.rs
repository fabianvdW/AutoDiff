use std::fmt::{Display, Formatter, Result};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Id(pub usize);
impl  Display for Id{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f,"{}",self.0)
    }
}