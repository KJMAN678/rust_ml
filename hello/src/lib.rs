pub fn hello_world(word: &str) -> String {
  let greeding :String = "Hello, ".to_owned() + word + "!";
  return greeding
}