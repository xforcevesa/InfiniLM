use crate::InferenceArgs;
use colored::Colorize;
use service::{Service, Session};
use std::{collections::HashMap, io::Write};
use transformer::SampleArgs;

impl InferenceArgs {
    pub async fn chat(self) {
        let mut chatting = Chatting::from(self);

        println!(
            "\
###########################################
# 欢迎使用九源推理框架-大模型单机对话demo #
###########################################"
        );
        chatting.print_args();
        println!();
        print_help();
        print_splitter();

        let mut input = String::new();
        loop {
            chatting.print_session();
            input.clear();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Unable to read line.");
            let input = input.trim();
            if !input.is_empty() {
                // 以 / 开头则为用户指令
                if input.starts_with('/') {
                    if !chatting.execute_command(input) {
                        break;
                    }
                } else {
                    chatting.infer(input).await;
                }
            }
        }
    }
}

struct Chatting {
    service: Service,
    sample: SampleArgs,
    session: Session,
    sessions: HashMap<usize, Session>,
}

impl From<InferenceArgs> for Chatting {
    fn from(args: InferenceArgs) -> Self {
        let service: Service = args.into();
        let session = service.launch();
        let sample = service.sample_args();
        Self {
            service,
            sample,
            session,
            sessions: HashMap::new(),
        }
    }
}

macro_rules! print_now {
    ($($arg:tt)*) => {{
        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}
fn print_splitter() {
    println!("=====================================");
}
fn print_help() {
    println!(
        "\
/create         新建会话session
/switch [0-9+]  切换至指定会话
/drop [0-9+]    丢弃指定会话
/args           打印当前参数
/args key value 设置指定参数
/help           打印帮助信息

使用 /exit 或 Ctrl + C 结束程序"
    );
}

impl Chatting {
    fn print_args(&self) {
        println!(
            "PID = {}, temperature = {}, top-k = {}, top-p = {}",
            std::process::id(),
            self.sample.temperature,
            self.sample.top_k,
            self.sample.top_p,
        );
    }

    fn print_session(&mut self) {
        print_now!(
            "{}{}{}",
            "User[".yellow(),
            self.session.id(),
            "]: ".yellow()
        );
    }

    fn execute_command(&mut self, command: &str) -> bool {
        match command.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["/create"] => {
                let old = std::mem::replace(&mut self.session, self.service.launch());
                self.sessions.insert(old.id(), old);
            }
            ["/switch", n] => match n.parse() {
                Ok(target_id) => {
                    if target_id == self.session.id() {
                        println!("Already in session {}", target_id);
                    } else if let Some(target) = self.sessions.remove(&target_id) {
                        let old = std::mem::replace(&mut self.session, target);
                        self.sessions.insert(old.id(), old);
                    } else {
                        println!("Invalid session ID.");
                    }
                }
                Err(_) => println!("Invalid drop command"),
            },
            ["/drop", n] => match n.parse() {
                Ok(target_id) => {
                    if target_id == self.session.id() {
                        if let Some((&id, _)) = self.sessions.iter().next() {
                            let _ = std::mem::replace(
                                &mut self.session,
                                self.sessions.remove(&id).unwrap(),
                            );
                        } else {
                            self.session = self.service.launch();
                        }
                        println!("Session {target_id} is dropped.")
                    } else if self.sessions.remove(&target_id).is_some() {
                        println!("Session {target_id} is dropped.");
                    } else {
                        println!("Invalid session ID.");
                    }
                }
                Err(_) => println!("Invalid drop command"),
            },
            ["/args"] => self.print_args(),
            ["/args", "temperature", t] => {
                if let Ok(t) = t.parse() {
                    self.sample.temperature = t;
                    self.service.set_sample_args(self.sample.clone());
                } else {
                    println!("Invalid temperature");
                }
            }
            ["/args", "top-k", k] => {
                if let Ok(k) = k.parse() {
                    self.sample.top_k = k;
                    self.service.set_sample_args(self.sample.clone());
                } else {
                    println!("Invalid top-k");
                }
            }
            ["/args", "top-p", p] => {
                if let Ok(p) = p.parse() {
                    self.sample.top_p = p;
                    self.service.set_sample_args(self.sample.clone());
                } else {
                    println!("Invalid top-p");
                }
            }
            ["/help"] => print_help(),
            ["/exit"] => return false,
            _ => println!("Unknown Command"),
        }
        print_splitter();
        true
    }

    async fn infer(&mut self, text: &str) {
        print_now!("{}", "AI: ".green());
        self.session
            .chat(text, |s| match s {
                "\\n" => println!(),
                _ => print_now!("{s}"),
            })
            .await;
        println!();
    }
}
