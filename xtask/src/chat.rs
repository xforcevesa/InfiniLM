use crate::{init_log, InferenceArgs};
use causal_lm::CausalLM;
use colored::Colorize;
use service::{Service, Session};
use std::{collections::HashMap, io::Write};

impl InferenceArgs {
    pub async fn chat(self) {
        init_log(self.log.as_deref());

        let (service, _handle) = Service::<transformer_cpu::Transformer>::new(self.model);
        let sessions = HashMap::from([(0, service.launch())]);
        Chatting {
            service,
            current: 0,
            next_id: 1,
            sessions,
        }
        .chat()
        .await;
    }
}

struct Chatting<M: CausalLM> {
    service: Service<M>,
    current: usize,
    next_id: usize,
    sessions: HashMap<usize, Session<M>>,
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

impl<M: CausalLM> Chatting<M> {
    async fn chat(mut self) {
        println!(
            "\
###########################################
# 欢迎使用九源推理框架-大模型单机对话demo #
###########################################"
        );
        self.print_args();
        println!();
        print_help();
        print_splitter();

        let mut input = String::new();
        loop {
            self.print_session();
            input.clear();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Unable to read line.");
            let input = input.trim();
            if !input.is_empty() {
                // 以 / 开头则为用户指令
                if input.starts_with('/') {
                    if !self.execute_command(input) {
                        break;
                    }
                } else {
                    self.infer(input).await;
                }
            }
        }
    }

    #[inline]
    fn session(&self) -> &Session<M> {
        self.sessions.get(&self.current).unwrap()
    }

    #[inline]
    fn session_mut(&mut self) -> &mut Session<M> {
        self.sessions.get_mut(&self.current).unwrap()
    }

    fn print_args(&self) {
        println!("PID = {}", std::process::id());
        println!("Current session = {}", self.current);
        let args = &self.session().sample;
        println!("temperature = {}", args.temperature);
        println!("top-k = {}", args.top_k);
        println!("top-p = {}", args.top_p);
    }

    fn print_session(&mut self) {
        print_now!("{}{}{}", "User[".yellow(), self.current, "]: ".yellow());
    }

    fn execute_command(&mut self, command: &str) -> bool {
        match command.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["/create"] => {
                self.current = self.next_id;
                self.next_id += 1;
                self.sessions.insert(self.current, self.service.launch());
            }
            ["/switch", n] => match n.parse() {
                Ok(target_id) => {
                    if target_id == self.current {
                        println!("Already in session {}", target_id);
                    } else if self.sessions.contains_key(&target_id) {
                        self.current = target_id;
                    } else {
                        println!("Invalid session ID.");
                    }
                }
                Err(_) => println!("Invalid drop command"),
            },
            ["/drop", n] => match n.parse() {
                Ok(target_id) => {
                    if self.sessions.remove(&target_id).is_none() {
                        println!("Invalid session ID.");
                    } else {
                        println!("Session {target_id} is dropped.");
                        if target_id == self.current {
                            if let Some((&id, _)) = self.sessions.iter().next() {
                                self.current = id;
                                println!("Current session is dropped, switched to {id}.");
                            } else {
                                self.current = self.next_id;
                                self.next_id += 1;
                                println!(
                                    "Current session is dropped, switched to new session {}.",
                                    self.current
                                );
                            }
                        }
                    }
                }
                Err(_) => println!("Invalid drop command"),
            },
            ["/args"] => self.print_args(),
            ["/args", "temperature", t] => {
                if let Ok(t) = t.parse() {
                    self.session_mut().sample.temperature = t;
                } else {
                    println!("Invalid temperature");
                }
            }
            ["/args", "top-k", k] => {
                if let Ok(k) = k.parse() {
                    self.session_mut().sample.top_k = k;
                } else {
                    println!("Invalid top-k");
                }
            }
            ["/args", "top-p", p] => {
                if let Ok(p) = p.parse() {
                    self.session_mut().sample.top_p = p;
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
        let mut busy = self.session_mut().chat([text]);
        while let Some(s) = busy.decode().await {
            match &*s {
                "\\n" => println!(),
                _ => print_now!("{s}"),
            }
        }
        println!();
    }
}
