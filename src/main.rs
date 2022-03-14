extern crate hello;

fn main() {
  let word:&str = "World";
  println!("{:?}", hello::hello_world(word));
}
